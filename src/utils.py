"""Shared utilities: chat template, tokenizer setup, collate, masking, and evaluation."""

import os

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ message['content'] }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Picture {{ image_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Video {{ video_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ content['text'] }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
}
EOS_TOKEN = "<|im_end|>"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def load_tokenizer(
    model_name: str = DEFAULT_TOKENIZER,
    padding_side: str = "right",
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def configure_model_tokens(model: PreTrainedModel, tokenizer: AutoTokenizer) -> None:
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


def configure_models_tokens(
    models: list[PreTrainedModel],
    tokenizer: AutoTokenizer,
) -> None:
    for model in models:
        configure_model_tokens(model, tokenizer)


def collate_fn(batch):
    filtered = [item for item in batch if item != 0]
    if len(filtered) == 0:
        return None
    keys = filtered[0].keys()
    return {key: torch.stack([item[key] for item in filtered]) for key in keys}


def assistant_mask_with_eos(
    input_ids: torch.Tensor,
    assistant_masks: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    mod_assistant_mask = assistant_masks.clone()
    matches = input_ids == tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    indices = torch.nonzero(matches)
    mod_assistant_mask[tuple(indices[-1])] = 1
    return mod_assistant_mask


def get_reward_client() -> OpenAI:
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def get_reward_score(prompt: str, response: str, client: OpenAI | None = None) -> float:
    if client is None:
        client = get_reward_client()
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    result = client.chat.completions.create(model=REWARD_MODEL, messages=messages)
    content = result.choices[0].message.content.strip()
    return float(content.split(":")[-1])


def compute_winrate(
    eval_prompts: list[str],
    model_a_completions: list[str],
    model_b_completions: list[str],
    client=None,
    prefer_a_on_tie: bool = False,
) -> tuple[float, dict[str, list[float]]]:
    if client is None:
        client = get_reward_client()

    wins = 0
    scores = {"a": [], "b": []}
    for prompt, response_a, response_b in tqdm(
        zip(eval_prompts, model_a_completions, model_b_completions),
        total=len(eval_prompts),
    ):
        reward_a = get_reward_score(prompt, response_a, client=client)
        reward_b = get_reward_score(prompt, response_b, client=client)
        scores["a"].append(reward_a)
        scores["b"].append(reward_b)
        if reward_a > reward_b or (prefer_a_on_tie and reward_a == reward_b):
            wins += 1

    return wins / len(eval_prompts), scores


def extract_user_prompt(messages: list[dict]) -> str | None:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return None

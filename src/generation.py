"""Generation utilities for HuggingFace, vLLM, and OpenAI batch API."""

import json
import os
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from src.data.score import extract_prompts, load_data
from src.utils import DEFAULT_TOKENIZER, load_tokenizer


def generate_batch(
    batch_size: int,
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_type: str = "instruct",
    max_new_tokens: int = 590,
) -> list[str]:
    rep_penalty = 1.0 if model_type == "instruct" else 1 + 1e-5
    outputs_list = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            repetition_penalty=rep_penalty,
            max_new_tokens=max_new_tokens,
        )
        completions_only = output_sequences[:, inputs["input_ids"].shape[1] :]
        outputs_list.extend(
            tokenizer.batch_decode(completions_only, skip_special_tokens=True)
        )
    return outputs_list


def generate_batch_dpo_style(
    batch_size: int,
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_type: str = "sft",
    max_new_tokens: int = 1024,
) -> list[str]:
    rep_penalty = 1.15 if model_type == "dpo" else 1 + 1e-5
    outputs_list = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            tokenizer=tokenizer,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=rep_penalty,
            stop_strings="<|im_end|>",
            exponential_decay_length_penalty=(int(max_new_tokens * 0.8), 1.0001),
            max_new_tokens=max_new_tokens,
        )
        completions_only = output_sequences[:, inputs["input_ids"].shape[1] :]
        outputs_list.extend(
            tokenizer.batch_decode(completions_only, skip_special_tokens=True)
        )
    return outputs_list


def student_generate_batch(
    batch_size: int,
    prompts: list[str],
    model: PreTrainedModel,
    max_tokens: int = 512,
    repetition_penalty: float = 1.12,
) -> list[str]:
    tokenizer = load_tokenizer(DEFAULT_TOKENIZER, padding_side="left")
    outputs_list = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            tokenizer=tokenizer,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            stop_strings="<|im_end|>",
            exponential_decay_length_penalty=(int(max_tokens * 0.7), 1.1),
            max_new_tokens=max_tokens,
        )
        completions_only = output_sequences[:, inputs["input_ids"].shape[1] :]
        outputs_list.extend(
            tokenizer.batch_decode(completions_only, skip_special_tokens=True)
        )
    return outputs_list


def generate_initial_responses(
    model_path: str,
    train_short: list[dict[str, Any]],
    train_med: list[dict[str, Any]],
    train_long: list[dict[str, Any]],
    test_raw: list[dict[str, Any]],
    output_dir: str = ".",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    max_tokens: int = 250,
    repetition_penalty: float = 1.22,
    stop_token_ids: list[int] | None = None,
) -> None:
    if stop_token_ids is None:
        stop_token_ids = [151465]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        stop_token_ids=stop_token_ids,
        device="cuda",
    )

    splits = {
        "short": load_data(train_short),
        "med": load_data(train_med),
        "long": load_data(train_long),
        "test": load_data(test_raw),
    }

    for split_name, formatted in splits.items():
        prompts = extract_prompts(formatted)
        completions = llm.generate(prompts, sampling_params)
        responses = [
            {"prompt": output.prompt, "completion": output.outputs[0].text}
            for output in completions
        ]
        output_path = f"{output_dir}/{split_name}_initial_responses.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(responses)} responses to {output_path}")


def truncate_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_input_tokens: int = 600,
) -> str:
    tokens = tokenizer(prompt)["input_ids"]
    if len(tokens) > max_input_tokens:
        tokens = tokens[:max_input_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return prompt


def teacher_generate_batch(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: str = "o4-mini-2025-04-16",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    input_path: str = "batch_input.jsonl",
) -> list[str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(input_path, "w") as f:
        for i, prompt in enumerate(prompts):
            item = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": truncate_prompt(prompt, tokenizer)},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stop": ["<|im_end|>"],
                    "frequency_penalty": 1.5,
                    "presence_penalty": 0.0,
                },
            }
            f.write(json.dumps(item) + "\n")

    with open(input_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print("Batch submitted. ID:", batch.id)

    while True:
        batch_status = client.batches.retrieve(batch.id)
        if batch_status.status in ["completed", "failed", "cancelled", "expired"]:
            break
        time.sleep(15)

    if batch_status.status != "completed":
        raise RuntimeError(f"Batch failed or didn't complete: {batch_status.status}")

    output_response = client.files.content(batch_status.output_file_id)
    responses = {}
    for line in output_response.text.splitlines():
        obj = json.loads(line)
        content = obj["response"]["body"]["choices"][0]["message"]["content"]
        responses[obj["custom_id"]] = content

    return [responses[f"request-{i}"] for i in range(len(prompts))]

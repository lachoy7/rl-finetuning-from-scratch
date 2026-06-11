"""SCoRe dataset preparation, formatting, and prompt construction."""

import json
from typing import Any

import torch
from sentence_transformers import SentenceTransformer, util

from src.data.score_categories import CATEGORIES_DICT, CRITERIA_DICT

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def assign_category(
    prompt: str,
    sentence_bert: SentenceTransformer | None = None,
    categories_dict: dict[str, list[str]] | None = None,
) -> str:
    if sentence_bert is None:
        sentence_bert = SentenceTransformer(DEFAULT_SBERT_MODEL)
    if categories_dict is None:
        categories_dict = CATEGORIES_DICT

    category_mean_embeds = {}
    for category, examples in categories_dict.items():
        example_embeds = sentence_bert.encode(examples, convert_to_tensor=True)
        category_mean_embeds[category] = torch.mean(example_embeds, dim=0)

    prompt_embed = sentence_bert.encode(prompt, convert_to_tensor=True)
    similarities = {
        category: util.cos_sim(prompt_embed, mean_embedding).item()
        for category, mean_embedding in category_mean_embeds.items()
    }
    return max(similarities, key=similarities.get)


def create_score_dataset(
    training_set_raw: list[str],
    sentence_bert: SentenceTransformer | None = None,
) -> list[dict[str, str]]:
    if sentence_bert is None:
        sentence_bert = SentenceTransformer(DEFAULT_SBERT_MODEL)

    return [
        {
            "x": instruction,
            "y": "",
            "c": CRITERIA_DICT[assign_category(instruction, sentence_bert=sentence_bert)],
        }
        for instruction in training_set_raw
    ]


def create_to_revise(x: str, c: str, r_0: str) -> str:
    return (
        "Below is an instruction and my initial response. A criteria for evaluating the response is also provided.\n\n"
        f"Instruction:\n{x}\n\n"
        f"My Initial Response:\n{r_0}\n\n"
        f"Criteria: {c}\n\n"
        "My initial response may be incorrect and may not follow the criteria. Please revise it using the ideal response as a guide and the criteria for improvement. "
        "Return only the revised answer, without any additional comments or explanation."
    )


def get_revisions(r_0_list: list[str], raw_data: list[dict[str, Any]]) -> list[str]:
    return [
        create_to_revise(item["x"], item["c"], r_0)
        for item, r_0 in zip(raw_data, r_0_list)
    ]


def load_data(train_raw: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    return [
        [
            {"content": str(item["x"]), "role": "user"},
            {"content": str(item["y"]), "role": "assistant"},
        ]
        for item in train_raw
    ]


def load_data_from_list(
    prompts: list[str],
    completions: list[str],
) -> list[list[dict[str, str]]]:
    return [
        [
            {"content": prompt, "role": "user"},
            {"content": completion, "role": "assistant"},
        ]
        for prompt, completion in zip(prompts, completions)
    ]


def extract_prompts(formatted_data: list[list[dict[str, str]]]) -> list[str]:
    return [item[0]["content"] for item in formatted_data]


def load_initial_responses(path: str) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        return []
    if isinstance(data[0], dict):
        return [item["completion"] for item in data]
    return data

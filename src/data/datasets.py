"""Dataset classes for SFT, DPO, and SCoRe training."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils import CHAT_TEMPLATE, assistant_mask_with_eos


class SFTSmolTalkDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length: int = 720):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        messages = self.dataset[idx]["messages"]
        new_messages = []
        for message in messages:
            if not new_messages and message["role"] == "user":
                new_messages.append(message)
            elif new_messages and message["role"] == "assistant":
                new_messages.append(message)
                break
        if len(new_messages) != 2:
            return 0

        try:
            tokenized = self.tokenizer.apply_chat_template(
                new_messages,
                tokenize=True,
                max_length=self.max_length,
                padding="max_length",
                truncation="only_second",
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
                chat_template=CHAT_TEMPLATE,
                return_tensors="pt",
            )
        except Exception:
            return 0

        input_ids = tokenized["input_ids"]
        mod_assistant_mask = assistant_mask_with_eos(
            input_ids, tokenized["assistant_masks"], self.tokenizer
        )
        labels = input_ids.clone()
        labels[mod_assistant_mask == 0] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class UltraFeedbackDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        chosen = example["chosen"]
        rejected = example["rejected"]
        if not len(chosen) == len(rejected) == 2:
            return 0

        try:
            chosen_tokenized = self.tokenizer.apply_chat_template(
                chosen,
                tokenize=True,
                max_length=self.max_len,
                padding="max_length",
                truncation="only_second",
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
                chat_template=CHAT_TEMPLATE,
                return_tensors="pt",
            )
            rejected_tokenized = self.tokenizer.apply_chat_template(
                rejected,
                tokenize=True,
                max_length=self.max_len,
                padding="max_length",
                truncation="only_second",
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
                chat_template=CHAT_TEMPLATE,
                return_tensors="pt",
            )
        except Exception:
            return 0

        chosen_input_ids = chosen_tokenized["input_ids"]
        rejected_input_ids = rejected_tokenized["input_ids"]

        return {
            "chosen_input_ids": chosen_input_ids.squeeze(0),
            "chosen_attention_mask": chosen_tokenized["attention_mask"].squeeze(0),
            "chosen_assistant_masks": assistant_mask_with_eos(
                chosen_input_ids,
                chosen_tokenized["assistant_masks"],
                self.tokenizer,
            ).squeeze(0),
            "rejected_input_ids": rejected_input_ids.squeeze(0),
            "rejected_attention_mask": rejected_tokenized["attention_mask"].squeeze(0),
            "rejected_assistant_masks": assistant_mask_with_eos(
                rejected_input_ids,
                rejected_tokenized["assistant_masks"],
                self.tokenizer,
            ).squeeze(0),
        }


class ScoreDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        messages = self.dataset[idx]
        new_messages = []
        for message in messages:
            if not new_messages and message["role"] == "user":
                new_messages.append(message)
            elif new_messages and message["role"] == "assistant":
                new_messages.append(message)
                break
        if len(new_messages) != 2:
            return self._empty_item()

        try:
            tokenized = self.tokenizer.apply_chat_template(
                new_messages,
                tokenize=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
                chat_template=CHAT_TEMPLATE,
                return_tensors="pt",
            )
        except Exception as exc:
            print(idx, exc)
            return self._empty_item()

        input_ids = tokenized["input_ids"]
        assistant_masks = tokenized["assistant_masks"]
        if assistant_masks.sum() == 0:
            return self._empty_item()

        mod_assistant_mask = assistant_mask_with_eos(
            input_ids, assistant_masks, self.tokenizer
        )
        labels = input_ids.clone()
        labels[mod_assistant_mask == 0] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _empty_item(self):
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.int32),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.int32),
            "labels": torch.full((self.max_length,), -100, dtype=torch.int32),
        }

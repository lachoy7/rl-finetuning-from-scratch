"""Evaluate SFT model vs reference using Nemotron reward win-rate."""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation import generate_batch
from src.utils import SPECIAL_TOKENS, compute_winrate, extract_user_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model win-rate")
    parser.add_argument("--model-path", default="./nonlora_sft_smoltalk_9k_b8_lr1e-5_ga32")
    parser.add_argument("--reference", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=590)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_raw = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
    eval_prompts = [
        extract_user_prompt(example["messages"])
        for example in test_raw.select(range(args.num_samples))
        if extract_user_prompt(example["messages"])
    ]

    sft_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sft_tokenizer.padding_side = "left"
    sft_tokenizer.add_special_tokens(SPECIAL_TOKENS)
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    sft_model.config.pad_token_id = sft_tokenizer.pad_token_id
    sft_model.config.eos_token_id = sft_tokenizer.eos_token_id
    sft_model.config.bos_token_id = sft_tokenizer.bos_token_id
    sft_model.eval()

    ref_tokenizer = AutoTokenizer.from_pretrained(args.reference)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.reference, torch_dtype=torch.float16
    ).to(device)
    ref_tokenizer.padding_side = "left"
    ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id
    ref_model.eval()

    sft_completions = generate_batch(
        args.batch_size, eval_prompts, sft_model, sft_tokenizer,
        model_type="sft", max_new_tokens=args.max_new_tokens,
    )
    ref_completions = generate_batch(
        args.batch_size, eval_prompts, ref_model, ref_tokenizer,
        model_type="instruct", max_new_tokens=args.max_new_tokens,
    )

    winrate, scores = compute_winrate(eval_prompts, sft_completions, ref_completions)
    print(f"SFT win-rate vs reference: {winrate:.4f}")
    print(f"Mean SFT reward: {sum(scores['a']) / len(scores['a']):.4f}")
    print(f"Mean reference reward: {sum(scores['b']) / len(scores['b']):.4f}")


if __name__ == "__main__":
    main()

"""Evaluate DPO model vs SFT baseline using Nemotron reward win-rate."""

import argparse
import pickle

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from src.generation import generate_batch_dpo_style
from src.utils import compute_winrate, configure_models_tokens, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DPO model win-rate")
    parser.add_argument("--sft-model", default="./sft_smoltalk_model")
    parser.add_argument("--dpo-model", default="./temp_dpo_model_epoch2")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sft-completions-pickle", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_raw = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
    eval_prompts = test_raw.select(range(args.num_samples))["prompt"]

    tokenizer = load_tokenizer(args.tokenizer, padding_side="left")
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model, torch_dtype=torch.float16
    ).to(device)
    dpo_model = AutoModelForCausalLM.from_pretrained(
        args.dpo_model, torch_dtype=torch.float16
    ).to(device)
    configure_models_tokens([sft_model, dpo_model], tokenizer)
    sft_model.eval()
    dpo_model.eval()

    if args.sft_completions_pickle:
        with open(args.sft_completions_pickle, "rb") as f:
            sft_completions = pickle.load(f)
    else:
        sft_completions = generate_batch_dpo_style(
            args.batch_size, eval_prompts, sft_model, tokenizer,
            model_type="sft", max_new_tokens=args.max_new_tokens,
        )

    dpo_completions = generate_batch_dpo_style(
        args.batch_size, eval_prompts, dpo_model, tokenizer,
        model_type="dpo", max_new_tokens=args.max_new_tokens,
    )

    winrate, scores = compute_winrate(
        eval_prompts, dpo_completions, sft_completions, prefer_a_on_tie=True,
    )
    print(f"DPO win-rate vs SFT: {winrate:.4f}")
    print(f"Mean DPO reward: {sum(scores['a']) / len(scores['a']):.4f}")
    print(f"Mean SFT reward: {sum(scores['b']) / len(scores['b']):.4f}")


if __name__ == "__main__":
    main()

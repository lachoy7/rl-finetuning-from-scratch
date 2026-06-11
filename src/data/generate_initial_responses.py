"""Generate initial student responses for SCoRe curriculum splits via vLLM."""

import argparse
import json

from src.generation import generate_initial_responses


def parse_args():
    parser = argparse.ArgumentParser(description="Generate initial SCoRe responses")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--repetition-penalty", type=float, default=1.22)
    parser.add_argument("--stop-token-ids", type=int, nargs="+", default=[151465])
    return parser.parse_args()


def main():
    args = parse_args()

    with open(f"{args.data_dir}/train_short.json") as f:
        train_short = json.load(f)
    with open(f"{args.data_dir}/train_med.json") as f:
        train_med = json.load(f)
    with open(f"{args.data_dir}/train_long.json") as f:
        train_long = json.load(f)
    with open(f"{args.data_dir}/test.json") as f:
        test_raw = json.load(f)

    generate_initial_responses(
        model_path=args.model_path,
        train_short=train_short,
        train_med=train_med,
        train_long=train_long,
        test_raw=test_raw,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=args.stop_token_ids,
    )


if __name__ == "__main__":
    main()

"""Prepare SCoRe leaderboard dataset with category assignment."""

import argparse
import json

import pandas as pd

from src.data.score import create_score_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SCoRe dataset from prompts")
    parser.add_argument(
        "--input",
        default="leaderboard_subs.jsonl",
        help="Input JSONL with a 'prompt' column",
    )
    parser.add_argument(
        "--output",
        default="leaderboard_raw.json",
        help="Output JSON file with x/y/c records",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_json(args.input, lines=True)
    dataset = create_score_dataset(df["prompt"].tolist())

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved {len(dataset)} records to {args.output}")


if __name__ == "__main__":
    main()

"""SCoRe curriculum learning training."""

import argparse
import json
import os
import pickle

import torch
from transformers import AutoModelForCausalLM

from src.data.score import load_initial_responses
from src.train.logging import add_wandb_args, log_config_from_args
from src.train.training import run_score
from src.utils import configure_model_tokens, load_tokenizer

STAGE_DEFAULTS = {
    "short": {"train_json": "train_short.json", "epochs": 13},
    "med": {"train_json": "train_med.json", "epochs": 8},
    "long": {"train_json": "train_long.json", "epochs": 5},
}


def parse_args():
    parser = argparse.ArgumentParser(description="SCoRe curriculum learning training")
    parser.add_argument("--student-checkpoint", default="./dpo_model")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--stage",
        choices=["short", "med", "long", "all"],
        default="all",
    )
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--test-prompts", default="test_prompts.pkl")
    parser.add_argument("--test-completions", default="test_completions.pkl")
    add_wandb_args(parser)
    return parser.parse_args()


def _load_json_list(path: str, max_samples: int | None) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def _load_revisions(stage: str) -> list[str]:
    primary = json.load(open(f"{stage}_revisions.json"))
    secondary_path = f"{stage}_revisions_2.json"
    if os.path.exists(secondary_path):
        primary.extend(json.load(open(secondary_path)))
    return primary


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.tokenizer, padding_side="right")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_checkpoint, torch_dtype=torch.float32
    )
    student.resize_token_embeddings(len(tokenizer))
    configure_model_tokens(student, tokenizer)
    student.to(device)

    with open(args.test_prompts, "rb") as f:
        test_prompts = pickle.load(f)
    with open(args.test_completions, "rb") as f:
        test_completions = pickle.load(f)

    stages = ["short", "med", "long"] if args.stage == "all" else [args.stage]
    for stage in stages:
        defaults = STAGE_DEFAULTS[stage]
        train_json = args.train_json or defaults["train_json"]
        epochs = args.epochs or defaults["epochs"]

        train_raw = _load_json_list(train_json, args.max_samples)
        r_0_list = load_initial_responses(f"{stage}_initial_responses.json")[
            : args.max_samples
        ]
        train_completions = _load_revisions(stage)

        print(f"Running SCoRe stage={stage}, epochs={epochs}, samples={len(train_raw)}")
        log_config = log_config_from_args(args, {"stage": "score", "score_stage": stage})
        if not log_config.run_name:
            log_config.run_name = f"score-{stage}"
        log_config.tags = [*args.wandb_tags, "score", stage]

        train_loss, val_loss = run_score(
            train_raw,
            r_0_list,
            train_completions,
            test_prompts,
            test_completions,
            student,
            tokenizer,
            device,
            epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            log_config=log_config,
        )
        print(f"Stage {stage} complete. Train loss: {train_loss}, Val loss: {val_loss}")


if __name__ == "__main__":
    main()

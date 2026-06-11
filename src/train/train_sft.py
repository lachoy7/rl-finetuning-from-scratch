"""Supervised fine-tuning on HuggingFaceTB/smol-smoltalk."""

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.data.datasets import SFTSmolTalkDataset
from src.train.logging import add_wandb_args, log_config_from_args
from src.train.training import fine_tune_sft
from src.utils import collate_fn, configure_model_tokens, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training on smol-smoltalk")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--resume-checkpoint", default="./sft_smoltalk_e1_18k")
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--train-start", type=int, default=300000)
    parser.add_argument("--train-end", type=int, default=455000)
    parser.add_argument("--val-start", type=int, default=455000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=720)
    parser.add_argument("--checkpoint-dir", default="./smoltalk")
    parser.add_argument("--output-dir", default="./sft_smoltalk_e1_28k")
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.tokenizer, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(args.resume_checkpoint)
    configure_model_tokens(model, tokenizer)
    model.to(device)

    train_loaded = load_dataset(args.dataset, split="train")
    train_dataset = SFTSmolTalkDataset(
        train_loaded.select(range(args.train_start, args.train_end)),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = SFTSmolTalkDataset(
        train_loaded.select(range(args.val_start, len(train_loaded))),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    log_config = log_config_from_args(args, {"stage": "sft"})
    if not log_config.run_name:
        log_config.run_name = "sft"

    fine_tune_sft(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        accumulation_steps=args.accum_steps,
        log_config=log_config,
    )
    model.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()

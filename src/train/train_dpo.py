"""DPO training on UltraFeedback binarized preferences."""

import argparse
import os

import torch
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.data.datasets import UltraFeedbackDataset
from src.train.training import fine_tune_dpo
from src.utils import collate_fn, configure_models_tokens, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training on UltraFeedback")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--base-checkpoint", default="./sft_smoltalk_e1_28k")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument(
        "--dataset-revision",
        default="292c16329d921287c4166934cac1a6ad1e13a6c5",
    )
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--accum-steps", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-dir", default="./final_dpo_modelv2")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer, padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint)
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint)
    configure_models_tokens([model, ref_model], tokenizer)

    train_loaded = load_dataset(
        args.dataset,
        revision=args.dataset_revision,
        split=args.split,
    )
    train_dataset = UltraFeedbackDataset(
        train_loaded.select(range(args.train_size)),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = UltraFeedbackDataset(
        train_loaded.select(range(args.train_size, len(train_loaded))),
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
    model.to(device)
    ref_model.to(device)

    fine_tune_dpo(
        model,
        ref_model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.beta,
        accumulation_steps=args.accum_steps,
    )
    model.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()

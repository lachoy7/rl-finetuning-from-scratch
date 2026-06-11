"""Training loops for SFT, DPO, and SCoRe curriculum learning."""

import math
import pickle
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.score import get_revisions, load_data_from_list
from src.data.datasets import ScoreDataset
from src.train.logging import LogConfig, TrainLogger


# --- SFT ---


def train_sft(
    model: PreTrainedModel,
    train_loader,
    optimizer,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
    accumulation_steps: int = 16,
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
    optimizer.zero_grad()

    for i, batch in enumerate(progress):
        start = time.time()
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / accumulation_steps
        assert not math.isnan(loss.item()), f"Loss: {loss}, Outputs: {outputs}"
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / (i + 1)

        if i % 50 == 0 and i != 0:
            writer.add_scalar("Loss/train", avg_loss, epoch * len(train_loader) + i)

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (len(train_loader) - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item() * accumulation_steps, avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if i % 1000 == 0 and i != 0:
            torch.cuda.empty_cache()
        if i % 100000 == 0 and i != 0:
            model.save_pretrained(f"./sft_model_e{epoch}_{i}")

    return total_loss / len(train_loader)


def test_sft(
    model: PreTrainedModel,
    test_loader,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=True)

    for i, batch in enumerate(progress):
        start = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if torch.all(labels == -100):
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        assert not math.isnan(loss.item()), f"Loss: {loss}, Outputs: {outputs}"

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (len(test_loader) - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item(), avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if i % 5000 == 0 and i != 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(test_loader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    return avg_loss


def fine_tune_sft(
    model: PreTrainedModel,
    train_loader,
    test_loader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: str = "./smoltalk",
    accumulation_steps: int = 16,
    log_config: LogConfig | None = None,
) -> None:
    writer = TrainLogger(log_config)

    for epoch in range(num_epochs):
        train_loss = train_sft(
            model,
            train_loader,
            optimizer,
            writer,
            epoch,
            device,
            accumulation_steps=accumulation_steps,
        )
        model.save_pretrained(checkpoint_dir)
        val_loss = test_sft(model, test_loader, writer, epoch, device)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        print(f"Epoch: {epoch}. Train Loss: {train_loss}. Val Loss: {val_loss}.")

    writer.close()


# --- DPO ---


def dpo_loss_fn(
    pi_logps_chosen,
    ref_logps_chosen,
    pi_logps_rejected,
    ref_logps_rejected,
    beta: float,
):
    log_ratio_chosen = pi_logps_chosen - ref_logps_chosen
    log_ratio_rejected = pi_logps_rejected - ref_logps_rejected
    return -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()


def _compute_logps(model, input_ids, attention_mask, assistant_masks):
    labels = input_ids.clone()
    labels[~assistant_masks.bool()] = 0
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logps = torch.gather(
        logits.log_softmax(dim=-1),
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return logps * assistant_masks


def train_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    train_loader,
    optimizer,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
    beta: float,
    accumulation_steps: int = 128,
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
    num_trained_examples = 0
    n = len(train_loader)
    optimizer.zero_grad()

    for i, batch in enumerate(progress):
        start = time.time()
        if batch is None:
            continue

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        if torch.all(chosen_input_ids == 0):
            continue

        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_assistant_masks = batch["chosen_assistant_masks"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_assistant_masks = batch["rejected_assistant_masks"].to(device)
        num_trained_examples += 1

        with torch.no_grad():
            ref_chosen_logps = _compute_logps(
                ref_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_assistant_masks,
            )
            ref_rejected_logps = _compute_logps(
                ref_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_assistant_masks,
            )

        pi_chosen_logps = _compute_logps(
            model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_assistant_masks,
        )
        pi_rejected_logps = _compute_logps(
            model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_assistant_masks,
        )

        loss = dpo_loss_fn(
            pi_chosen_logps,
            ref_chosen_logps,
            pi_rejected_logps,
            ref_rejected_logps,
            beta,
        )
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == n:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / num_trained_examples

        if i % 10 == 0 and i != 0:
            writer.add_scalar("Loss/train", avg_loss, epoch * n + i)

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (n - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item() * accumulation_steps, avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if (i % 1000 == 0 and i != 0) or i == len(train_loader):
            torch.cuda.empty_cache()

    return total_loss / num_trained_examples


def test_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    test_loader,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
    beta: float,
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=True)
    num_tested_examples = 0

    for i, batch in enumerate(progress):
        start = time.time()
        if batch is None:
            continue

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        if torch.all(chosen_input_ids == 0):
            continue

        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_assistant_masks = batch["chosen_assistant_masks"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_assistant_masks = batch["rejected_assistant_masks"].to(device)
        num_tested_examples += 1

        with torch.no_grad():
            ref_chosen_logps = _compute_logps(
                ref_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_assistant_masks,
            )
            ref_rejected_logps = _compute_logps(
                ref_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_assistant_masks,
            )
            pi_chosen_logps = _compute_logps(
                model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_assistant_masks,
            )
            pi_rejected_logps = _compute_logps(
                model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_assistant_masks,
            )
            loss = dpo_loss_fn(
                pi_chosen_logps,
                ref_chosen_logps,
                pi_rejected_logps,
                ref_rejected_logps,
                beta,
            )

        total_loss += loss.item()
        avg_loss = total_loss / num_tested_examples

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (len(test_loader) - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item(), avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if (i % 500 == 0 and i != 0) or i == len(test_loader):
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_tested_examples
    writer.add_scalar("Loss/val", avg_loss, epoch)
    return avg_loss


def fine_tune_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    train_loader,
    test_loader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    beta: float,
    accumulation_steps: int = 128,
    log_config: LogConfig | None = None,
) -> None:
    writer = TrainLogger(log_config)

    for epoch in range(num_epochs):
        train_loss = train_dpo(
            model,
            ref_model,
            train_loader,
            optimizer,
            writer,
            epoch,
            device,
            beta,
            accumulation_steps=accumulation_steps,
        )
        val_loss = test_dpo(
            model, ref_model, test_loader, writer, epoch, device, beta
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        print(f"Epoch: {epoch}. Train Loss: {train_loss}. Val Loss: {val_loss}.")
        model.save_pretrained(f"./dpo_epoch{epoch}_v2")

    writer.close()


# --- SCoRe ---


def train_score(
    model: PreTrainedModel,
    train_loader,
    optimizer,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
    accumulation_steps: int = 8,
    checkpoint_dir: str = "./latest_model",
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
    optimizer.zero_grad()

    for i, batch in enumerate(progress):
        start = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if torch.all(labels == -100) or torch.all(input_ids == 0):
            print(f"Skipping empty batch {i}")
            continue

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / accumulation_steps
        if torch.isnan(loss):
            continue
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / (i + 1)
        writer.add_scalar("Loss/train", avg_loss, epoch * len(train_loader) + i)

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (len(train_loader) - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item() * accumulation_steps, avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if i % 1000 == 0 and i != 0:
            torch.cuda.empty_cache()
            model.save_pretrained(checkpoint_dir)
            with open("latest_opt.pkl", "wb") as f:
                pickle.dump(optimizer, f)

    return float(total_loss / len(train_loader))


def test_score(
    model: PreTrainedModel,
    test_loader,
    writer: TrainLogger,
    epoch: int,
    device: torch.device,
) -> float:
    total_loss = 0
    batch_times = []
    progress = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=True)

    for i, batch in enumerate(progress):
        start = time.time()
        input_ids = batch["input_ids"].long().to(device)
        attention_mask = batch["attention_mask"].long().to(device)
        labels = batch["labels"].long().to(device)

        if torch.all(labels == -100) or torch.all(input_ids == 0):
            print(f"Skipping empty batch {i}")
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        loss = outputs.loss
        if math.isnan(loss):
            print("NAN loss")
            continue

        total_loss += loss.item()
        avg_loss = float(total_loss) / (i + 1)

        batch_time = time.time() - start
        batch_times.append(batch_time)
        avg_time = sum(batch_times) / len(batch_times)
        eta = avg_time * (len(test_loader) - (i + 1))
        eta_hr, remainder = divmod(int(eta), 3600)
        eta_min, eta_sec = divmod(remainder, 60)
        progress.set_postfix(
            loss=[loss.item(), avg_loss],
            eta=f"{eta_hr}h {eta_min}m {eta_sec}s",
        )

        if (i + 1) % 10 == 0:
            model.save_pretrained("./checkpoints/latest_step")
        if i % 1000 == 0 and i != 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(test_loader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    return float(avg_loss)


def fine_tune_score(
    model: PreTrainedModel,
    train_loader,
    test_loader,
    optimizer,
    device: torch.device,
    num_epochs: int,
    log_config: LogConfig | None = None,
) -> tuple[float, float]:
    writer = TrainLogger(log_config)
    train_loss = val_loss = 0.0

    for epoch in range(num_epochs):
        train_loss = train_score(model, train_loader, optimizer, writer, epoch, device)
        val_loss = test_score(model, test_loader, writer, epoch, device)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        print(f"Epoch: {epoch}. Train Loss: {train_loss}. Val Loss: {val_loss}.")

    writer.close()
    return train_loss, val_loss


def run_score(
    train_raw: list,
    r_0_list: list[str],
    train_completions: list[str],
    test_prompts: list[str],
    test_completions: list[str],
    student: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    num_epochs: int,
    batch_size: int = 32,
    lr: float = 1e-7,
    log_config: LogConfig | None = None,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
    )

    train_prompts = get_revisions(r_0_list, train_raw)
    train_set = ScoreDataset(load_data_from_list(train_prompts, train_completions), tokenizer)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_set = ScoreDataset(load_data_from_list(test_prompts, test_completions), tokenizer)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return fine_tune_score(
        model=student,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        log_config=log_config,
    )

"""Training metrics logging for TensorBoard and Weights & Biases."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from torch.utils.tensorboard import SummaryWriter


@dataclass
class LogConfig:
    project: str = "rl-finetuning"
    run_name: str | None = None
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] | None = None
    use_wandb: bool = True
    use_tensorboard: bool = True
    tensorboard_dir: str | None = None


class TrainLogger:
    """Logs scalars to TensorBoard and/or Weights & Biases."""

    def __init__(self, log_config: LogConfig | None = None):
        log_config = log_config or LogConfig()
        self._tb_writer: SummaryWriter | None = None
        self._wandb_run = None

        if log_config.use_tensorboard:
            self._tb_writer = SummaryWriter(log_config.tensorboard_dir)

        if log_config.use_wandb and os.environ.get("WANDB_MODE") != "disabled":
            import wandb

            self._wandb_run = wandb.init(
                project=log_config.project,
                name=log_config.run_name,
                entity=log_config.entity,
                tags=log_config.tags or None,
                config=log_config.config,
                reinit=True,
            )

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)
        if self._wandb_run is not None:
            import wandb

            wandb.log({tag: value}, step=step)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()


def add_wandb_args(parser) -> None:
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "rl-finetuning"),
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME"),
        help="Weights & Biases run name (optional)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Weights & Biases entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=[],
        help="Weights & Biases tags",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )


def log_config_from_args(args, extra_config: dict[str, Any] | None = None) -> LogConfig:
    config = dict(extra_config or {})
    for key, value in vars(args).items():
        if key.startswith("wandb_") or key in {"no_wandb", "no_tensorboard"}:
            continue
        config[key] = value
    return LogConfig(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        tags=args.wandb_tags,
        config=config,
        use_wandb=not args.no_wandb,
        use_tensorboard=not args.no_tensorboard,
    )

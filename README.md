# RL Finetuning from Scratch

Class project for **Deep Reinforcement Learning**, Spring 2025.

Fine-tune **Qwen2.5-0.5B** through supervised fine-tuning (SFT), direct preference optimization (DPO), and [SCoRe](https://arxiv.org/abs/2409.12917)-style self-correction curriculum learning. This repo implements the full pipeline from initial response generation through training and reward-model evaluation.

The self-correction stage is inspired by **SCoRe** (Self-Correction via Reinforcement Learning), which trains language models to revise their own outputs using multi-turn learning and criteria-guided feedback. See Kumar et al., [*Training Language Models to Self-Correct via Reinforcement Learning*](https://arxiv.org/abs/2409.12917) (arXiv:2409.12917).

Base model: `Qwen/Qwen2.5-0.5B`

## Setup

### 1. Install micromamba

[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a fast, standalone conda-compatible package manager. If you do not have it yet, follow the official [Micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Quick install (Linux, macOS, or Git Bash on Windows):

```bash
"${SHELL}" <(curl -L https://micro.mamba.pm/install.sh)
```

On macOS you can also use Homebrew:

```bash
brew install micromamba
```

After installing, restart your shell or run `micromamba shell init` so `micromamba activate` works in new terminals. See the [Micromamba user guide](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html) for shell setup details.

### 2. Create the environment

From the repo root, create and activate the `rl-finetuning` environment:

```bash
micromamba create -f environment.yml
micromamba activate rl-finetuning
pip install -r requirements.txt
```

Or create the environment manually without `environment.yml`:

```bash
micromamba create -n rl-finetuning -c conda-forge python=3.11 pip
micromamba activate rl-finetuning
pip install -r requirements.txt
```

**Optional — Linux with NVIDIA GPU:** install PyTorch with CUDA from conda before running `pip install`, so training and vLLM can use the GPU:

```bash
micromamba install -n rl-finetuning -c pytorch -c nvidia pytorch pytorch-cuda=12.1
pip install -r requirements.txt
```

Adjust the CUDA version (`12.1`, `11.8`, etc.) to match your driver. See [PyTorch install instructions](https://pytorch.org/get-started/locally/) if needed.

### 3. API keys

Set API keys as needed:

| Variable | Used for |
|----------|----------|
| `HF_TOKEN` | Downloading gated HuggingFace datasets |
| `NVIDIA_API_KEY` | Nemotron reward model evaluation |
| `OPENAI_API_KEY` | Teacher batch generation during SCoRe (optional) |
| `WANDB_API_KEY` | Weights & Biases experiment tracking (optional) |

## Project layout

```
environment.yml             # Micromamba env definition (Python 3.11)
requirements.txt            # Pip dependencies
src/
  utils.py                    # Chat template, tokenizer setup, collate, reward scoring
  generation.py               # HuggingFace, vLLM, and OpenAI batch generation
  data/
    score_categories.py       # Category criteria and example prompts (static data)
    score.py                    # SCoRe data prep, prompts, formatting
    datasets.py                 # SFT, DPO, and SCoRe dataset classes
    prepare_score_data.py       # Assign criteria to leaderboard prompts
    generate_initial_responses.py
  train/
    logging.py                  # TensorBoard and Weights & Biases logging
    training.py                 # SFT, DPO, and SCoRe training loops
    train_sft.py
    train_dpo.py
    train_score.py
  eval/
    eval_sft.py
    eval_dpo.py
scripts/                        # Shell wrappers for each stage
```

## Pipeline

Run stages in order:

```
generate initial responses → prepare SCoRe data → SFT → DPO → SCoRe → eval
```

### 1. Generate initial responses (vLLM, GPU required)

Produces `short/med/long/test_initial_responses.json` from curriculum split JSON files.

```bash
MODEL_PATH=./dpo_model ./scripts/generate_initial_responses.sh
```

### 2. Prepare SCoRe leaderboard data

Assigns evaluation criteria to prompts via SBERT category matching.

```bash
./scripts/prepare_score_data.sh
```

Requires `leaderboard_subs.jsonl` with a `prompt` column. Writes `leaderboard_raw.json`.

### 3. SFT on smol-smoltalk

```bash
./scripts/train_sft.sh
```

Defaults: resume from `./sft_smoltalk_e1_18k`, train indices 300k–455k, save to `./sft_smoltalk_e1_28k`.

### 4. DPO on UltraFeedback

```bash
HF_TOKEN=your_token ./scripts/train_dpo.sh
```

Defaults: start from SFT checkpoint, 60k train examples, 2 epochs, save to `./final_dpo_modelv2`.

### 5. SCoRe self-correction training

Trains short (13 epochs) → med (8) → long (5) curriculum stages sequentially, following the multi-turn self-correction idea from [SCoRe](https://arxiv.org/abs/2409.12917).

```bash
./scripts/train_score.sh
```

Requires pre-generated initial responses, teacher revisions (`*_revisions.json`), and test pickles (`test_prompts.pkl`, `test_completions.pkl`).

Train a single stage:

```bash
STAGE=short ./scripts/train_score.sh
```

### 6. Evaluation

Compare models using the NVIDIA Nemotron-70B reward model:

```bash
NVIDIA_API_KEY=your_key ./scripts/eval_sft.sh
NVIDIA_API_KEY=your_key ./scripts/eval_dpo.sh
```

## Running directly

Each script has a matching Python module with CLI flags:

```bash
python -m src.train.train_sft --help
python -m src.train.train_dpo --help
python -m src.train.train_score --stage short --epochs 13
python -m src.eval.eval_sft --model-path ./my_sft_model
python -m src.eval.eval_dpo --dpo-model ./my_dpo_model
```

Shell scripts accept environment variables to override defaults (e.g. `OUTPUT_DIR`, `BATCH_SIZE`, `LR`, `MAX_SAMPLES`).

### Weights & Biases

Training scripts log `Loss/train`, `Loss/val`, and `Loss/train_epoch` to both TensorBoard and [Weights & Biases](https://wandb.ai) by default. Set `WANDB_API_KEY` (or run `wandb login`) before training.

```bash
export WANDB_API_KEY=your_key
export WANDB_PROJECT=rl-finetuning   # optional, this is the default
./scripts/train_sft.sh
```

Disable wandb with `--no-wandb` or `NO_WANDB=1`. Override run metadata via `--wandb-run-name`, `--wandb-entity`, or the `WANDB_RUN_NAME` / `WANDB_ENTITY` environment variables.

## Data files

The SCoRe stage expects these files in the working directory:

| File | Description |
|------|-------------|
| `train_short.json`, `train_med.json`, `train_long.json` | Curriculum splits with `x`, `y`, `c` fields |
| `test.json` | Test split |
| `{short,med,long}_initial_responses.json` | Student initial completions |
| `{short,med,long}_revisions.json` | Teacher revised completions |
| `test_prompts.pkl`, `test_completions.pkl` | Test set for SCoRe training loop |

Initial response JSON can be a plain string list or `{prompt, completion}` dicts.

## Reference

Kumar, A., Zhuang, V., Agarwal, R., et al. (2024). *Training Language Models to Self-Correct via Reinforcement Learning*. arXiv:2409.12917. https://arxiv.org/abs/2409.12917

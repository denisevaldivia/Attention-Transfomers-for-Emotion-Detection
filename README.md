# AITA — Moral Judgment LLM

Fine-tuning an open-source LLM to act as an impartial moral judge on interpersonal conflicts, using Reddit's r/AmITheAsshole community as a training signal.

## Overview

This project fine-tunes **Qwen2.5-7B-Instruct** with **LoRA** to classify first-person conflict descriptions into one of four verdicts: **NTA**, **YTA**, **ESH**, or **NAH**. The model is trained on 5,000 curated examples sourced from a 270k-post Reddit dataset, with Claude Haiku used as a preprocessing assistant to transform noisy Reddit posts into clean, structured training samples.

## Dataset & Processing

Raw posts were filtered by upvote score (≥100 for NTA/YTA, ≥10 for ESH/NAH), length, and comment quality, then balanced to 1,250 samples per class. Claude Haiku rewrote each post into a concise first-person situation summary and synthesized the top two community comments into a unified moral judgment, with strict 200/160-word limits and explicit tone constraints to avoid hedging language.

## Model & Training

LoRA (rank 16) was used to fine-tune on a single A100 GPU, training only a few million parameters instead of the full 7B. Training ran for 3 epochs with train and validation loss staying closely aligned throughout. The checkpoint at step 400 was selected as the best balance between task learning and natural language quality.

## Results & Limitations

The model handles clear-cut cases well and produces coherent, direct reasoning. It inherits Reddit's demographic biases — filling unstated details with statistically common patterns from the training data — and reflects community consensus rather than universal ethics. With 5,000 examples it generalizes well to common conflict patterns but struggles with unusual or heavily overlapping situations.

## Stack

`Qwen2.5-7B-Instruct` · `LoRA / PEFT` · `TRL / SFTTrainer` · `Claude Haiku` · `Gradio` · `Hugging Face Datasets`

```plaintext
aita-moral-judgment/
├── Notebooks/
│   ├── 01_dataset_preparation.ipynb   # filtering, cleaning, Haiku preprocessing
│   ├── 02_training.ipynb              # LoRA fine-tuning with SFTTrainer
│   ├── 03_evaluation.ipynb            # held-out eval, confusion matrix, error analysis
│   └── 04_demo.ipynb                  # Gradio inference UI
├── Dataset/
│   ├── aita_sample_seed.txt           # fixed post IDs used for training
│   └── aita_finetune.jsonl
└── README.md
```
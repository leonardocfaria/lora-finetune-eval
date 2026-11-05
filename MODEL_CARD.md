# Model card: docstring-module classifier (GPT-2-arch 1.9 M + LoRA r=8)

## Model details
- **Architecture:** GPT-2 causal LM (`transformers.GPT2LMHeadModel`): 4 layers, d_model 192, 6 heads, context 512, byte-level vocabulary of 258 (256 bytes + pad + eos), 1.93 M parameters.
- **Base model:** pretrained from scratch in this repo (`run.py`, seed 0) for 1500 steps of causal LM on 378 k characters of Python 3.12 standard-library docstrings from 37 modules. Final training loss 1.50 nats/byte. No external weights.
- **Adapter:** PEFT LoRA, r=8, α=16, dropout 0.05, on `c_attn`, `c_proj`, `c_fc`. It has 98,304 trainable parameters (4.9 %) and was trained 8 epochs with AdamW, lr 2e-3, batch 16, with loss on the label tokens only.
- **Artifacts:** `results/export/adapter` (397 KB, safetensors) and `results/export/merged` (adapter merged into the base, 7.7 MB).
- **License:** code MIT-style per repo; training text from CPython docstrings (PSF License).

## Intended use
- A teaching and benchmark artifact. It shows a complete parameter-efficient fine-tuning workflow on CPU: tracking, base-vs-tuned evaluation, merge/export.
- **Not for production use**, and not a general language model.

## Task and data
- **Input:** `Docstring: <first 300 chars of a docstring>\nModule:`.
- **Output:** one of `os, math, statistics, typing, asyncio, logging`, chosen by the highest mean token log-likelihood of ` <label>\n`.
- **Data:**
  - docstrings of public functions and classes of those 6 modules, at most 60 per module, deduplicated;
  - stratified 50/50 split with seed 0: 129 train / 131 test;
  - these modules are excluded from pretraining.

## Evaluation (held-out test, n = 131)
| model | accuracy | macro-F1 |
|---|---|---|
| base, zero-shot | 0.137 | 0.134 |
| + LoRA r=8 (this model) | 0.443 | 0.428 |
| full fine-tune (reference) | 0.458 | 0.470 |

Chance = 0.167. 95 % CI on accuracy ≈ ±8.5 points (n = 131).

## Limitations and biases
- Tiny model and tiny data: 44 % accuracy is far from useful. It is meaningful only relative to the base model and to full fine-tuning.
- Skewed predictions: the model over-predicts `os` (51 of 131), the largest class in training.
- Results depend on the Python version (docstrings change between releases) and come from a single seed. The hyper-parameters were not tuned on a validation split.
- Byte-level tokens and English technical text only.

## How to reproduce
`python run.py` (about 16 min on 4 vCPUs). Every run's configuration and metrics are written to `results/runs.jsonl`.

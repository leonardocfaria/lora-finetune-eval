# lora-finetune-eval

LoRA fine-tuning of a small open causal LM on CPU, with PEFT. It covers:
- a tiny task subset;
- tracked runs;
- base vs tuned evaluation on a held-out set;
- a LoRA rank ablation and a full fine-tune reference;
- adapter + merged-model export;
- a [model card](MODEL_CARD.md).

## Setup
```bash
cd ml/lora-finetune-eval
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q       # 4 tests, ~1 min (tiny model)
python run.py --quick     # smoke run of the whole pipeline, ~20 s
python run.py             # full run, ~16 min on 4 vCPUs (13 of them pretraining the base model)
```
Outputs:
- `results/runs.jsonl`: one line per run (config, trainable params, loss curve, metrics, time);
- `results/eval.md`: the table below;
- `results/base/`: the base model;
- `results/export/adapter/`: the LoRA adapter, safetensors;
- `results/export/merged/`: the merged full model.

Model weights are git-ignored; rerun to get them.

## Architecture
```
stdlib docstrings of 37 modules (PSF) ─► byte-level causal LM pretraining ─► base model (GPT-2 arch, 1.93 M params)
                                                                                │
6 *other* modules' docstrings ─► "Docstring: …\nModule:" → " math" (129 train / 131 test, stratified, seed 0)
                                                                                │
             zero-shot base │ PEFT LoRA r=2, r=8 (c_attn, c_proj, c_fc; α = 2r; loss on label tokens only) │ full FT
                                                                                │
             evaluation: argmax over the 6 labels of the mean token log-likelihood of " <label>\n"
                                                                                │
             best adapter ─► save_pretrained (adapter) ─► merge_and_unload ─► save_pretrained (merged)
```
Why a self-pretrained base model:
- Hugging Face is blocked from this sandbox, so no pretrained checkpoint (distilgpt2, tiny Qwen, …) can be downloaded.
- The base is therefore a standard `transformers.GPT2LMHeadModel`, pretrained here on Python stdlib docstrings. That makes it a fully open model with a reproducible recipe.
- **The fine-tuning task uses six modules never seen in pretraining.** A test checks that test texts do not occur in the pretraining corpus.

With a real checkpoint, swap `base_model()` for `AutoModelForCausalLM.from_pretrained(...)` and a real tokenizer. The LoRA, evaluation and export code stay as they are.

Tracking is one JSON line per run. That is enough for a handful of CPU runs; MLflow tracking is shown in `ml/mlops-tabular-e2e`.

## Results (seed 0, 4 vCPUs)
| run | trainable params | train s | test accuracy | macro-F1 |
|---|---|---|---|---|
| base, zero-shot | - | - | 0.137 | 0.134 |
| LoRA r=2 | 24,576 (1.3 %) | 68 | 0.321 | 0.241 |
| **LoRA r=8** | 98,304 (4.9 %) | 55 | **0.443** | **0.428** |
| full fine-tune | 1,927,680 (100 %) | 42 | 0.458 | 0.470 |
| merged LoRA r=8 | - | - | 0.443 | 0.428 |

- Chance is 1/6 = 0.167. The base model is at chance zero-shot: it never saw the "Module:" format or these modules.
- **LoRA r=8 trains 4.9 % of the parameters and reaches 97 % of full fine-tuning's accuracy.** With 131 test examples the 95 % CI is about ±8.5 points, so r=8 and full fine-tuning are statistically tied. r=2 is clearly worse.
- LoRA over-predicts `os` (51 of 131 predictions, against 30 true `os` test examples). Full fine-tuning spreads its predictions more evenly, hence its higher macro-F1.
- **Merging is exact:** the merged model gives identical predictions. A test checks that merged logits equal adapter logits and that a reloaded adapter equals the in-memory one.
- **Export size:** the adapter is 397 KB vs 7.7 MB for the merged model, a 19× smaller artifact to ship per task.
- Loss curves are still falling at epoch 8 (r=8: 3.36 → 0.62), so more epochs or a learning-rate sweep on a validation split would likely help. That was not done, to avoid tuning on the test set.

## Not done (time box)
A real pretrained checkpoint, a validation split + hyper-parameter search, QLoRA/4-bit, more seeds, generation-quality evaluation.

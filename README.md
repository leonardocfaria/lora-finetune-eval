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
python -m pytest -q       # 5 tests, ~10 s (tiny model)
python run.py --quick     # smoke run of the whole pipeline, ~20 s
python run.py             # full run, ~20 min on 4 vCPUs (17 of them pretraining the base model)
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
6 *other* modules' docstrings ─► "Docstring: …\nModule:" → " math" (121 train / 123 test, stratified, seed 0)
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
- **The fine-tuning task uses six modules never seen in pretraining.**
  - Only objects *defined* in each module count (by `__module__`), so re-exports such as `statistics.Counter` are not mislabelled.
  - Any docstring that also appears in the pretraining corpus (e.g. the `typing.Type` alias carries `builtins.type`'s docstring) is dropped.
  - A test checks that no train or test text occurs in the pretraining corpus.
  - Pretraining text still mentions some label words in passing (e.g. "see os.chmod()"), a weak, unavoidable link.

With a real checkpoint, swap `base_model()` for `AutoModelForCausalLM.from_pretrained(...)` and a real tokenizer. The LoRA, evaluation and export code stay as they are.

Tracking is one JSON line per run. That is enough for a handful of CPU runs; MLflow tracking is shown in `ml/mlops-tabular-e2e`.

## Results (seed 0, 4 vCPUs)
| run | trainable params | train s | test accuracy | macro-F1 |
|---|---|---|---|---|
| base, zero-shot | - | - | 0.106 | 0.119 |
| LoRA r=2 | 24,576 (1.3 %) | 53 | 0.293 | 0.244 |
| **LoRA r=8** (exported) | 98,304 (4.9 %) | 51 | **0.480** | **0.476** |
| full fine-tune | 1,927,680 (100 %) | 40 | 0.545 | 0.531 |
| merged LoRA r=8 | - | - | 0.480 | 0.476 |

- Chance is 1/6 = 0.167. The base model is below chance zero-shot: it never saw the "Module:" format or these modules, and it mostly answers `typing`.
- **LoRA r=8 trains 4.9 % of the parameters and reaches 88 % of full fine-tuning's accuracy** (0.480 vs 0.545). With 123 test examples the 95 % CI is about ±8.8 points, so the 6.5-point gap is not significant. r=2 is clearly worse.
- **Caveat:** the learning rates differ (LoRA 2e-3, full FT 3e-4) and neither was swept. The comparison is indicative, not tuned.
- The exported rank (r=8) is fixed in `run.py`, not picked by test accuracy.
- **Merging is exact:** the merged model gives identical predictions. A test checks that merged logits equal adapter logits and that a reloaded adapter equals the in-memory one.
- **Export size:** the adapter is 397 KB vs 7.7 MB for the merged model, a 19× smaller artifact to ship per task.
- **Small classes suffer.** `statistics` has only 9 test examples and is almost never predicted (4 of 123 predictions by LoRA, 2 by full FT).
- Loss curves are still falling at epoch 8 (r=8: 3.49 → 0.53), so more epochs or a learning-rate sweep on a validation split would likely help.

**Self-review fix.** The first version walked `dir(module)`, which includes imported names. 13 task docstrings were mislabelled (e.g. `collections.Counter` labelled "statistics"), and they also appeared verbatim in the pretraining corpus. The leak test only checked 30 rows, so it missed this. After the fix (filter by `__module__`, drop overlaps, test all rows) the results changed from 0.443/0.458 (LoRA r=8/full) to 0.480/0.545, and the README claim is now true.

## Not done (time box)
A real pretrained checkpoint, a validation split + hyper-parameter search, QLoRA/4-bit, more seeds, generation-quality evaluation.

# lora-finetune-eval Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A complete, honest LoRA workflow on CPU: base model → PEFT LoRA fine-tune on a tiny subset → tracked runs → base vs tuned evaluation on a held-out set → merged/exported model → model card.
- **Base model:** Hugging Face is blocked (DECISIONS #3), so no downloadable checkpoint exists. The "small open LM" is a GPT-2-architecture model (`transformers.GPT2LMHeadModel`, 4 layers, d=192, byte-level vocab, ~1.9 M params), **pretrained here** as a causal LM on Python stdlib docstrings (PSF licence) from ~25 modules. Its weights and recipe are fully open (this repo).
- **Task / dataset:** public, local, and disjoint from pretraining. Classify which stdlib module a docstring belongs to, among 6 modules never seen in pretraining: `os, math, statistics, typing, asyncio, logging`. Up to 60 docstrings per class, stratified 50/50 train/test split with a seed. Instruction format: `Docstring: …\nModule:` → ` math`.
- **Scoring:** for each candidate label, mean per-token log-likelihood of ` <label>\n` given the prompt; predict the argmax. Base model zero-shot vs LoRA-tuned: accuracy and macro-F1 on the held-out half.
- **Fine-tuning:** PEFT `LoraConfig(r, alpha=2r, target c_attn/c_proj/c_fc, dropout 0.05)`, loss on label tokens only (prompt masked with -100), AdamW, fixed seed. Ablation: r ∈ {2, 8}, plus a full fine-tune for reference.
- **Tracking:** one JSON line per run in `results/runs.jsonl` (config, trainable params, loss curve, metrics, seconds). That is enough for a handful of CPU runs; MLflow is shown in `mlops-tabular-e2e`.
- **Export:** the adapter (`save_pretrained`, safetensors, tiny), then `merge_and_unload()` → full merged model saved. A test checks that merged logits equal adapter-model logits.
- **Non-goals:** a real pretrained LLM, QLoRA/4-bit, generation quality.

## Files
- `lora/data.py`, `lora/model.py` (byte tokenizer, `base_model`, `pretrain`), `lora/finetune.py` (examples, `train_lora`, `train_full`, `evaluate`, `export`), `run.py`, `MODEL_CARD.md`, tests.

## Tasks (TDD)
- [x] 1. data disjointness/determinism; tokenizer round trip; prompt masking.
- [x] 2. LoRA: only adapter params train, base frozen and unchanged; loss drops.
- [x] 3. merge: merged logits == adapter logits; export files exist.
- [x] 4. run.py end to end (pretrain → base eval → LoRA r=2/8 → full FT → export); eval table; model card; README; CI.

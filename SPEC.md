# lora-finetune-eval

## Goal
Build "LoRA fine-tuning with evaluation and model card". Fine-tune a small open LM with LoRA (PEFT) on a public instruction or classification dataset, on CPU with a tiny subset, and track runs; evaluate against the base model on a held-out set; write a model card and merge/export the adapter. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] LoRA training runs on CPU with the subset
- [ ] base vs tuned eval table committed
- [ ] model card written
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open


"""Pretrain the base model, evaluate it zero-shot, LoRA-tune (r=2, r=8) and fully fine-tune, track every run,
export the r=8 adapter (adapter + merged). python run.py [--out results] [--quick]   (~20 min on 4 vCPUs)"""
import argparse
import json
import time
from pathlib import Path

import torch

from lora.data import LABELS, pretrain_corpus, task_split
from lora.finetune import evaluate, export, lora, train
from lora.model import base_model, pretrain


EXPORT_RANK = 8


def log_run(path: Path, run: dict):
    with path.open("a") as f:
        f.write(json.dumps(run) + "\n")
    print({k: v for k, v in run.items() if k != "loss_curve"})


def main(out="results", quick=False, seed=0):
    out = Path(out)
    out.mkdir(exist_ok=True)
    runs = out / "runs.jsonl"
    runs.unlink(missing_ok=True)
    train_rows, test_rows = task_split(seed)
    corpus = pretrain_corpus()

    base = base_model(seed) if not quick else base_model(seed, n_embd=64, n_layer=2, n_head=4)
    curve = pretrain(base, corpus, steps=60 if quick else 1500, seq=64 if quick else 256)
    base.save_pretrained(out / "base")
    log_run(runs, {"run": "pretrain", "corpus_chars": len(corpus), "params": sum(p.numel() for p in base.parameters()),
                   "loss_curve": curve, "seconds": round(base.pretrain_seconds, 1)})
    log_run(runs, {"run": "base zero-shot", **evaluate(base, test_rows)})

    epochs = 2 if quick else 8
    exported = None
    for r in (2, 8):
        m = lora(base, r=r, seed=seed)
        stats = train(m, train_rows, epochs=epochs, lr=2e-3, seed=seed)
        res = {"run": f"LoRA r={r}", "r": r, "lr": 2e-3, "epochs": epochs, **stats, **evaluate(m, test_rows)}
        log_run(runs, res)
        if r == EXPORT_RANK:  # fixed in advance: picking the best on the test set would tune on it
            exported = (m, res)
    full = base_model(seed) if not quick else base_model(seed, n_embd=64, n_layer=2, n_head=4)
    full.load_state_dict(base.state_dict())
    stats = train(full, train_rows, epochs=epochs, lr=3e-4, seed=seed)
    log_run(runs, {"run": "full fine-tune", "lr": 3e-4, "epochs": epochs, **stats, **evaluate(full, test_rows)})

    exp = export(exported[0], out / "export")
    merged_eval = evaluate(exp.pop("merged"), test_rows)
    log_run(runs, {"run": f"merged ({exported[1]['run']})", **exp, **merged_eval})

    rows = [json.loads(line) for line in runs.read_text().splitlines()]
    lines = [f"Task: which of {len(LABELS)} stdlib modules a docstring comes from ({', '.join(LABELS)}); "
             f"{len(train_rows)} train / {len(test_rows)} held-out test examples, chance = {1 / len(LABELS):.0%}.", "",
             "| run | trainable params | train s | accuracy | macro-F1 |", "|---|---|---|---|---|"]
    for r in rows[1:]:
        tp = f"{r['trainable_params']:,} ({r['trainable_params'] / r['total_params']:.1%})" if "trainable_params" in r else "-"
        lines.append(f"| {r['run']} | {tp} | {r.get('train_seconds', '-')} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    lines += ["", f"Base model: {rows[0]['params']:,} params, pretrained {rows[0]['seconds']} s on {rows[0]['corpus_chars']:,} chars; "
              f"loss {rows[0]['loss_curve'][0]} → {rows[0]['loss_curve'][-1]}.",
              f"Export: adapter {exp['adapter_bytes'] / 1e3:.0f} KB vs merged model {exp['merged_bytes'] / 1e6:.1f} MB."]
    (out / "eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    t = time.perf_counter()
    main(a.out, a.quick)
    print(f"total {time.perf_counter() - t:.0f} s")

"""LoRA (PEFT) and full fine-tuning on label tokens only; likelihood-based evaluation; adapter + merged export."""
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model

from .data import LABELS, prompt
from .model import EOS, PAD, encode


def example(doc: str, label: str) -> tuple[list[int], list[int]]:
    """input ids and labels; prompt positions are -100 so the loss only covers ' <label>\\n<eos>'."""
    p, target = encode(prompt(doc)), encode(f" {label}\n") + [EOS]
    return p + target, [-100] * len(p) + target


def batches(rows, batch_size, gen):
    order = torch.randperm(len(rows), generator=gen).tolist()
    for s in range(0, len(order), batch_size):
        exs = [example(*rows[i]) for i in order[s: s + batch_size]]
        n = max(len(ids) for ids, _ in exs)
        ids = torch.tensor([x + [PAD] * (n - len(x)) for x, _ in exs])
        lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in exs])
        yield ids, lab, (ids != PAD).long()


def train(model, rows, epochs=8, lr=2e-3, batch_size=16, seed=0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    curve, t = [], time.perf_counter()
    for _ in range(epochs):
        losses = []
        for ids, lab, mask in batches(rows, batch_size, gen):
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curve.append(round(sum(losses) / len(losses), 4))
    model.eval()
    return {"loss_curve": curve, "train_seconds": round(time.perf_counter() - t, 1),
            "trainable_params": sum(p.numel() for p in params),
            "total_params": sum(p.numel() for p in model.parameters())}


def lora(base, r: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=["c_attn", "c_proj", "c_fc"],
                     fan_in_fan_out=True, task_type="CAUSAL_LM")  # GPT-2 uses Conv1D (transposed) layers
    return get_peft_model(copy.deepcopy(base), cfg)


@torch.no_grad()
def label_scores(model, doc: str) -> list[float]:
    """Mean per-token log-likelihood of ' <label>\\n' after the prompt, for every label (length-normalised)."""
    p = encode(prompt(doc))
    conts = [encode(f" {lab}\n") for lab in LABELS]
    n = len(p) + max(map(len, conts))
    ids = torch.tensor([p + c + [PAD] * (n - len(p) - len(c)) for c in conts])
    logp = F.log_softmax(model(input_ids=ids, attention_mask=(ids != PAD).long()).logits, -1)
    out = []
    for k, c in enumerate(conts):
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(c))  # logits at t predict token t+1
        out.append(logp[k, pos, torch.tensor(c)].mean().item())
    return out


def evaluate(model, rows) -> dict:
    model.eval()
    preds = [LABELS[max(range(len(LABELS)), key=s.__getitem__)] for s in (label_scores(model, d) for d, _ in rows)]
    gold = [lab for _, lab in rows]
    f1s = []
    for lab in LABELS:
        tp = sum(p == g == lab for p, g in zip(preds, gold))
        fp = sum(p == lab != g for p, g in zip(preds, gold))
        fn = sum(g == lab != p for p, g in zip(preds, gold))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"accuracy": sum(p == g for p, g in zip(preds, gold)) / len(gold), "macro_f1": sum(f1s) / len(f1s),
            "n": len(gold), "predicted_label_counts": {lab: preds.count(lab) for lab in LABELS}}


def export(peft_model, out: str | Path) -> dict:
    """Save the adapter alone, then merge it into the base weights and save the full model (safetensors)."""
    out = Path(out)
    peft_model.save_pretrained(out / "adapter")
    merged = copy.deepcopy(peft_model).merge_and_unload()
    merged.save_pretrained(out / "merged")
    size = lambda d: sum(f.stat().st_size for f in (out / d).glob("*.safetensors"))
    return {"adapter_bytes": size("adapter"), "merged_bytes": size("merged"), "merged": merged}


def load_adapter(base, path):
    return PeftModel.from_pretrained(copy.deepcopy(base), path).eval()

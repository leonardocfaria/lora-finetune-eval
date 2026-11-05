"""Byte-level tokenizer and a small GPT-2 (transformers) pretrained here as the open base model."""
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

PAD, EOS = 256, 257
VOCAB = 258


def encode(text: str) -> list[int]:
    return list(text.encode())


def decode(ids) -> str:
    return bytes(i for i in ids if i < 256).decode(errors="replace")


def base_model(seed: int = 0, n_embd: int = 192, n_layer: int = 4, n_head: int = 6) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=512, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     bos_token_id=EOS, eos_token_id=EOS, pad_token_id=PAD)
    return GPT2LMHeadModel(cfg)


def pretrain(model, text: str, steps: int = 1500, seq: int = 256, batch: int = 16, lr: float = 1e-3, seed: int = 0,
             log_every: int = 100) -> list[float]:
    """Causal-LM pretraining on random windows of the corpus. Returns the loss curve (every log_every steps)."""
    data = torch.tensor(encode(text))
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    curve, t = [], time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, len(data) - seq - 1, (batch,), generator=g)
        x = torch.stack([data[i: i + seq] for i in idx])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append(round(loss.item(), 4))
    model.eval()
    model.pretrain_seconds = time.perf_counter() - t
    return curve

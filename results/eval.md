Task: which of 6 stdlib modules a docstring comes from (os, math, statistics, typing, asyncio, logging); 121 train / 123 held-out test examples, chance = 17%.

| run | trainable params | train s | accuracy | macro-F1 |
|---|---|---|---|---|
| base zero-shot | - | - | 0.106 | 0.119 |
| LoRA r=2 | 24,576 (1.3%) | 52.5 | 0.293 | 0.244 |
| LoRA r=8 | 98,304 (4.9%) | 51.3 | 0.480 | 0.476 |
| full fine-tune | 1,927,680 (100.0%) | 39.7 | 0.545 | 0.531 |
| merged (LoRA r=8) | - | - | 0.480 | 0.476 |

Base model: 1,927,680 params, pretrained 1035.2 s on 316,008 chars; loss 5.5072 → 1.5475.
Export: adapter 397 KB vs merged model 7.7 MB.

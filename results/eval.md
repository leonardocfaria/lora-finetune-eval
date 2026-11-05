Task: which of 6 stdlib modules a docstring comes from (os, math, statistics, typing, asyncio, logging); 129 train / 131 held-out test examples, chance = 17%.

| run | trainable params | train s | accuracy | macro-F1 |
|---|---|---|---|---|
| base zero-shot | - | - | 0.137 | 0.134 |
| LoRA r=2 | 24,576 (1.3%) | 67.6 | 0.321 | 0.241 |
| LoRA r=8 | 98,304 (4.9%) | 55.2 | 0.443 | 0.428 |
| full fine-tune | 1,927,680 (100.0%) | 42.4 | 0.458 | 0.470 |
| merged (LoRA r=8) | - | - | 0.443 | 0.428 |

Base model: 1,927,680 params, pretrained 774.5 s on 378,236 chars; loss 5.5634 → 1.5008.
Export: adapter 397 KB vs merged model 7.7 MB.

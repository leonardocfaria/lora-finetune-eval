import pytest
import torch

from lora.data import LABELS, PRETRAIN_MODULES, docstrings, pretrain_corpus, task_split
from lora.finetune import evaluate, example, export, label_scores, load_adapter, lora, train
from lora.model import base_model, decode, encode, pretrain


def test_data_disjoint_deterministic_and_stratified():
    assert not set(LABELS) & set(PRETRAIN_MODULES)
    train_rows, test_rows = task_split(0)
    assert (train_rows, test_rows) == task_split(0)
    assert not {d for d, _ in train_rows} & {d for d, _ in test_rows}
    assert {lab for _, lab in test_rows} == set(LABELS)
    corpus = pretrain_corpus()
    assert len(corpus) > 100_000
    assert not any(d in corpus for d, _ in test_rows[:30])  # task texts are not in pretraining


def test_tokenizer_and_label_only_loss_mask():
    assert decode(encode("héllo")) == "héllo"
    ids, labels = example("Return the cosine of x.", "math")
    n_target = len(encode(" math\n")) + 1
    assert labels[:-n_target] == [-100] * (len(ids) - n_target) and labels[-n_target:] == ids[-n_target:]


@pytest.fixture(scope="module")
def tiny():
    base = base_model(n_embd=64, n_layer=2, n_head=4)
    pretrain(base, pretrain_corpus()[:50_000], steps=30, seq=64, batch=8)
    train_rows, test_rows = task_split(0)
    return base, train_rows[:48], test_rows[:24]


def test_lora_trains_only_adapters_and_leaves_base_untouched(tiny):
    base, train_rows, _ = tiny
    before = {k: v.clone() for k, v in base.state_dict().items()}
    m = lora(base, r=4)
    stats = train(m, train_rows, epochs=3)
    assert stats["loss_curve"][-1] < stats["loss_curve"][0]
    assert stats["trainable_params"] < 0.1 * stats["total_params"]
    assert all(("lora_" in n) == p.requires_grad for n, p in m.named_parameters())
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())  # deepcopy: base unchanged


def test_merge_matches_adapter_and_export_roundtrip(tiny, tmp_path):
    base, train_rows, test_rows = tiny
    m = lora(base, r=4)
    train(m, train_rows, epochs=1)
    with torch.no_grad():  # make the adapter non-trivial (B starts at 0) so merging actually changes weights
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.01)
    out = export(m, tmp_path)
    assert out["adapter_bytes"] < out["merged_bytes"] / 5
    doc = test_rows[0][0]
    assert label_scores(out["merged"], doc) == pytest.approx(label_scores(m, doc), abs=1e-4)
    reloaded = load_adapter(base, tmp_path / "adapter")
    assert label_scores(reloaded, doc) == pytest.approx(label_scores(m, doc), abs=1e-4)
    assert set(evaluate(out["merged"], test_rows)) >= {"accuracy", "macro_f1"}

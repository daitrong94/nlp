"""
LAB 09 - Dữ liệu CoNLL-2003 và metric cho NER (dùng chung cho baseline, notebook và IE).

- Dataset: CoNLL-2003 (bản parquet lhoestq/conll2003), lưu ở ../hf_models, load offline.
- Nhãn BIO: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
- Entity-level metric tự viết theo quy ước conlleval: một entity đúng khi đúng cả loại lẫn ranh giới (start, end).
"""

import glob
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
TYPES = ["PER", "ORG", "LOC", "MISC"]


def load_conll(split):
    """Trả về list các câu, mỗi câu là (words, tags) với tags là chuỗi BIO."""
    import pandas as pd
    root = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "datasets--lhoestq--conll2003", "snapshots", "*"))[0]
    df = pd.read_parquet(glob.glob(os.path.join(root, "data", f"{split}-*.parquet"))[0])
    return [(list(w), [LABELS[t] for t in tags]) for w, tags in zip(df["tokens"], df["ner_tags"]) if len(w) > 0]


def get_entities(tags):
    """BIO → tập entity (type, start, end), end không tính.
    Theo conlleval: I-X đứng sau O hoặc sau loại khác cũng mở một entity mới."""
    ents, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        prefix, _, cur = t.partition("-")
        # đóng entity đang mở nếu gặp O, gặp B-, hoặc gặp I- của loại khác
        if typ is not None and (prefix != "I" or cur != typ):
            ents.append((typ, start, i))
            typ = None
        if prefix == "B" or (prefix == "I" and typ is None):
            start, typ = i, cur
    return set(ents)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def entity_metrics(gold_seqs, pred_seqs):
    """Entity-level P/R/F1, micro trên toàn bộ entity và riêng từng loại."""
    c = {t: Counter() for t in TYPES + ["ALL"]}
    for g, p in zip(gold_seqs, pred_seqs):
        G, P = get_entities(g), get_entities(p)
        for e in G & P:
            c[e[0]]["tp"] += 1; c["ALL"]["tp"] += 1
        for e in P - G:
            c[e[0]]["fp"] += 1; c["ALL"]["fp"] += 1
        for e in G - P:
            c[e[0]]["fn"] += 1; c["ALL"]["fn"] += 1
    return {t: prf(v["tp"], v["fp"], v["fn"]) for t, v in c.items()}


def token_metrics(gold_seqs, pred_seqs):
    """Token-level: accuracy trên mọi token, và P/R/F1 trên các token không phải O (đúng khi đúng nguyên nhãn BIO)."""
    tp = fp = fn = correct = total = 0
    for g, p in zip(gold_seqs, pred_seqs):
        for a, b in zip(g, p):
            total += 1
            correct += a == b
            if b != "O" and a == b:
                tp += 1
            if b != "O" and a != b:
                fp += 1
            if a != "O" and a != b:
                fn += 1
    return {"accuracy": correct / total, **prf(tp, fp, fn)}


if __name__ == "__main__":
    # ví dụ trong đề: New York, gold B-LOC I-LOC, prediction B-LOC O
    gold, pred = ["B-LOC", "I-LOC"], ["B-LOC", "O"]
    assert get_entities(gold) == {("LOC", 0, 2)} and get_entities(pred) == {("LOC", 0, 1)}
    m = entity_metrics([gold], [pred])["ALL"]
    t = token_metrics([gold], [pred])
    print("New York: token accuracy", t["accuracy"], "| entity F1", m["f1"])
    assert t["accuracy"] == 0.5 and m["f1"] == 0.0
    assert get_entities(["B-PER", "I-PER", "O", "B-LOC", "B-LOC"]) == {("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 5)}
    assert get_entities(["O", "I-ORG", "I-ORG", "I-LOC"]) == {("ORG", 1, 3), ("LOC", 3, 4)}
    assert get_entities(["B-PER", "I-ORG"]) == {("PER", 0, 1), ("ORG", 1, 2)}
    tr = load_conll("train")
    print("train sentences:", len(tr), "| ví dụ:", list(zip(*tr[1])))
    assert all(len(w) == len(t) for w, t in tr)
    print("All tests passed.")

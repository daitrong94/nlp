"""
LAB 09 - Information Extraction đơn giản (Option A - News): raw text → NER → entity → JSON.

    "Barack Obama visited Paris."  →  {"PERSON": ["Barack Obama"], "LOCATION": ["Paris"]}

- Model: DistilBERT đã fine-tune trên CoNLL-2003 trong ner_finetuning.ipynb, lưu ở ../hf_models/lab9_ner_model.
- Đánh giá: 30 câu tin tức tự viết (2022 - 2026, khác thời kỳ với CoNLL-2003 là tin Reuters 1996), có gold annotation
  cho PERSON / ORGANIZATION / LOCATION. Tính entity-level P/R/F1: một entity đúng khi đúng cả loại lẫn chuỗi chữ.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_data import get_entities, prf

MODEL_DIR = os.path.join(os.environ["HF_HOME"], "lab9_ner_model")
NAMES = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}
TOKEN_RE = re.compile(r"(?:[A-Z]\.){2,}|'s\b|\w+(?:[-&]\w+)*|[^\w\s]")      # U.S. | 's | word, Coca-Cola | dấu câu


def split_words(text):
    """Tách raw text thành word theo kiểu CoNLL, giữ vị trí ký tự để lấy lại đúng chuỗi gốc."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class Extractor:
    def __init__(self, model_dir=MODEL_DIR):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def tag(self, words):
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
        pred = self.model(**enc).logits[0].argmax(-1).tolist()
        tags, prev = ["O"] * len(words), None
        for pos, w in enumerate(enc.word_ids(0)):
            if w is not None and w != prev:              # nhãn của word = nhãn ở token đầu (Strategy A)
                tags[w] = self.labels[pred[pos]]
            prev = w
        return tags

    def extract(self, text, keep_misc=False):
        """Raw text → {"PERSON": [...], "ORGANIZATION": [...], "LOCATION": [...]}."""
        spans = split_words(text)
        tags = self.tag([w for w, _, _ in spans])
        out = {NAMES[t]: [] for t in ["PER", "ORG", "LOC"] + (["MISC"] if keep_misc else [])}
        for typ, s, e in sorted(get_entities(tags), key=lambda x: x[1]):
            if NAMES[typ] in out:
                out[NAMES[typ]].append(text[spans[s][1]:spans[e - 1][2]])
        return out


# 30 câu tin tức + gold annotation (P = PERSON, O = ORGANIZATION, L = LOCATION)
GOLD = [
    ("Barack Obama visited Paris.", {"P": ["Barack Obama"], "L": ["Paris"]}),
    ("Elon Musk said Tesla will open a new factory near Berlin next year.", {"P": ["Elon Musk"], "O": ["Tesla"], "L": ["Berlin"]}),
    ("OpenAI chief executive Sam Altman met lawmakers in Washington on Tuesday.", {"P": ["Sam Altman"], "O": ["OpenAI"], "L": ["Washington"]}),
    ("Nvidia shares rose 5% after the company reported record revenue.", {"O": ["Nvidia"]}),
    ("President Volodymyr Zelenskiy thanked Germany and France for their support.", {"P": ["Volodymyr Zelenskiy"], "L": ["Germany", "France"]}),
    ("The World Health Organization warned of a new outbreak in the Democratic Republic of Congo.", {"O": ["World Health Organization"], "L": ["Democratic Republic of Congo"]}),
    ("Lionel Messi scored twice as Inter Miami beat Orlando City 3-1.", {"P": ["Lionel Messi"], "O": ["Inter Miami", "Orlando City"]}),
    ("Samsung Electronics plans to invest $17 billion in a chip plant in Taylor, Texas.", {"O": ["Samsung Electronics"], "L": ["Taylor", "Texas"]}),
    ("Vietnam's Prime Minister Pham Minh Chinh arrived in Tokyo for talks with Japanese officials.", {"P": ["Pham Minh Chinh"], "L": ["Vietnam", "Tokyo"]}),
    ("The European Central Bank kept interest rates unchanged, Christine Lagarde said in Frankfurt.", {"O": ["European Central Bank"], "P": ["Christine Lagarde"], "L": ["Frankfurt"]}),
    ("Apple and Google face new antitrust rules in the European Union.", {"O": ["Apple", "Google", "European Union"]}),
    ("Taylor Swift will perform three concerts in Singapore in March.", {"P": ["Taylor Swift"], "L": ["Singapore"]}),
    ("Researchers at Stanford University and MIT published the study in Nature.", {"O": ["Stanford University", "MIT", "Nature"]}),
    ("Jordan signed a trade agreement with Saudi Arabia on Sunday.", {"L": ["Jordan", "Saudi Arabia"]}),
    ("Michael Jordan sold his majority stake in the Charlotte Hornets.", {"P": ["Michael Jordan"], "O": ["Charlotte Hornets"]}),
    ("Amazon is cutting 9,000 jobs, chief executive Andy Jassy told staff.", {"O": ["Amazon"], "P": ["Andy Jassy"]}),
    ("Heavy rain flooded parts of the Amazon near Manaus, Brazil.", {"L": ["Amazon", "Manaus", "Brazil"]}),
    ("VinFast delivered its first electric cars to customers in California.", {"O": ["VinFast"], "L": ["California"]}),
    ("Carlos Alcaraz beat Novak Djokovic in the Wimbledon final in London.", {"P": ["Carlos Alcaraz", "Novak Djokovic"], "L": ["London"]}),
    ("The United Nations Security Council will meet in New York to discuss Gaza.", {"O": ["United Nations Security Council"], "L": ["New York", "Gaza"]}),
    ("Microsoft agreed to buy Activision Blizzard for $69 billion.", {"O": ["Microsoft", "Activision Blizzard"]}),
    ("Chancellor Olaf Scholz and President Emmanuel Macron spoke by phone on Friday.", {"P": ["Olaf Scholz", "Emmanuel Macron"]}),
    ("TSMC said production in Taiwan was not affected by the earthquake near Hualien.", {"O": ["TSMC"], "L": ["Taiwan", "Hualien"]}),
    ("Manchester City manager Pep Guardiola praised Erling Haaland after the win over Arsenal.", {"O": ["Manchester City", "Arsenal"], "P": ["Pep Guardiola", "Erling Haaland"]}),
    ("Thousands marched through Manchester on Saturday to protest the cuts.", {"L": ["Manchester"]}),
    ("The International Monetary Fund raised its growth forecast for India.", {"O": ["International Monetary Fund"], "L": ["India"]}),
    ("Anthropic and DeepMind researchers warned about risks from advanced AI systems.", {"O": ["Anthropic", "DeepMind"]}),
    ("NASA administrator Bill Nelson said the Artemis crew will launch from Florida.", {"O": ["NASA"], "P": ["Bill Nelson"], "L": ["Florida"]}),
    ("Hanoi and Ho Chi Minh City reported record tourist arrivals, the ministry said.", {"L": ["Hanoi", "Ho Chi Minh City"]}),
    ("tesla boss elon musk visits beijing to meet chinese officials", {"O": ["tesla"], "P": ["elon musk"], "L": ["beijing"]}),
]
KEY = {"P": "PERSON", "O": "ORGANIZATION", "L": "LOCATION"}


def evaluate(extractor, gold=GOLD, verbose=False):
    """Entity-level P/R/F1: so khớp theo (loại, chuỗi chữ) trong từng câu."""
    c = {t: Counter() for t in list(KEY.values()) + ["ALL"]}
    errors = []
    for text, g in gold:
        pred = extractor.extract(text)
        G = Counter((KEY[k], x) for k, v in g.items() for x in v)
        P = Counter((t, x) for t, v in pred.items() for x in v)
        for (t, x), n in (G & P).items():
            c[t]["tp"] += n; c["ALL"]["tp"] += n
        for (t, x), n in (P - G).items():
            c[t]["fp"] += n; c["ALL"]["fp"] += n
        for (t, x), n in (G - P).items():
            c[t]["fn"] += n; c["ALL"]["fn"] += n
        if P != G:
            errors.append({"text": text, "missing": sorted((G - P).elements()), "wrong": sorted((P - G).elements())})
        if verbose:
            print(text, "\n  →", json.dumps(pred, ensure_ascii=False))
    return {t: prf(v["tp"], v["fp"], v["fn"]) for t, v in c.items()}, errors


if __name__ == "__main__":
    assert [w for w, _, _ in split_words("Vietnam's U.S. envoy, Coca-Cola.")] == ["Vietnam", "'s", "U.S.", "envoy", ",", "Coca-Cola", "."]
    ex = Extractor()
    if len(sys.argv) > 1:                                # python information_extraction.py "some text"
        print(json.dumps(ex.extract(" ".join(sys.argv[1:]), keep_misc=True), indent=2, ensure_ascii=False))
        sys.exit()

    print(json.dumps(ex.extract("Barack Obama visited Paris."), indent=2))
    metrics, errors = evaluate(ex, verbose=True)
    n_ent = sum(len(v) for _, g in GOLD for v in g.values())
    print(f"\n{len(GOLD)} câu, {n_ent} gold entity")
    for t, m in metrics.items():
        print(f"{t:13s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")
    print(f"\nCâu có lỗi: {len(errors)}/{len(GOLD)}")
    for e in errors:
        print(" -", e["text"], "\n     thiếu:", e["missing"], "| thừa/sai:", e["wrong"])

    # ghi thêm vào results.csv
    import pandas as pd
    rows = [{"experiment": "information_extraction", "model": "Transformer", "setting": t, "split": "30 câu tin tức",
             "metric": k, "value": m[k]} for t, m in metrics.items() for k in ["precision", "recall", "f1", "tp", "fp", "fn"]]
    path = Path(__file__).with_name("results.csv")
    if path.exists():
        old = pd.read_csv(path)
        pd.concat([old[old.experiment != "information_extraction"], pd.DataFrame(rows)]).to_csv(path, index=False)
        print("đã ghi kết quả IE vào results.csv")

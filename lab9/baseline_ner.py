"""
LAB 09 - Baseline NER: token features → Logistic Regression.

Mỗi word được phân loại độc lập thành một trong 9 nhãn BIO, dựa trên:
- chính từ đó: dạng lowercase, word shape (Xxxx, XX, dd...), viết hoa, chữ số, gạch nối, prefix/suffix 1 - 3 ký tự;
- cửa sổ ±2 từ: lowercase và shape của các từ lân cận, đánh dấu đầu/cuối câu.

Không có pretraining, không có embedding và không nhìn được ngữ cảnh xa hơn 2 từ.
"""

import re
import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ner_data import entity_metrics, load_conll, token_metrics


def shape(w):
    """'Washington' → 'Xx', 'U.S.' → 'X.X.', '1996' → 'd', 'SpaceX' → 'XxX'."""
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1+", r"\1", s)                  # gộp ký tự lặp: Xxxxx → Xx


def word_features(words, i):
    w = words[i]
    f = {"bias": 1, "w": w.lower(), "shape": shape(w), "title": w.istitle(), "upper": w.isupper(),
         "digit": any(c.isdigit() for c in w), "hyphen": "-" in w, "first": i == 0,
         "len": min(len(w), 8)}
    for k in (1, 2, 3):
        f[f"pre{k}"] = w[:k].lower()
        f[f"suf{k}"] = w[-k:].lower()
    for d in (-2, -1, 1, 2):
        j = i + d
        if 0 <= j < len(words):
            f[f"w{d}"] = words[j].lower()
            f[f"shape{d}"] = shape(words[j])
        else:
            f[f"w{d}"] = "<BOS>" if j < 0 else "<EOS>"
    f["w-1|w"] = f["w-1"] + "|" + f["w"]                # cặp từ trước + từ hiện tại
    return f


def featurize(sents):
    return [word_features(words, i) for words, _ in sents for i in range(len(words))]


class BaselineNER:
    def __init__(self, C=1.0, max_iter=300):
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, sents):
        X = self.vec.fit_transform(featurize(sents))
        y = [t for _, tags in sents for t in tags]
        self.clf.fit(X, y)
        return self

    def predict(self, sents):
        flat = list(self.clf.predict(self.vec.transform(featurize(sents))))
        out, k = [], 0
        for words, _ in sents:
            out.append(flat[k:k + len(words)])
            k += len(words)
        return out

    def predict_proba(self, sents):
        """Xác suất của nhãn được chọn cho từng word (dùng cho phân tích confidence)."""
        P = self.clf.predict_proba(self.vec.transform(featurize(sents))).max(1)
        out, k = [], 0
        for words, _ in sents:
            out.append(P[k:k + len(words)])
            k += len(words)
        return out


if __name__ == "__main__":
    assert shape("Washington") == "Xx" and shape("U.S.") == "X.X." and shape("1996") == "d"
    train, val, test = load_conll("train"), load_conll("validation"), load_conll("test")
    t0 = time.perf_counter()
    model = BaselineNER().fit(train)
    print(f"train {time.perf_counter() - t0:.0f} s, {len(model.vec.vocabulary_):,} features")
    for name, data in [("validation", val), ("test", test)]:
        gold = [t for _, t in data]
        pred = model.predict(data)
        e, tk = entity_metrics(gold, pred), token_metrics(gold, pred)
        print(f"{name}: token acc {tk['accuracy']:.4f} | entity P {e['ALL']['precision']:.4f} "
              f"R {e['ALL']['recall']:.4f} F1 {e['ALL']['f1']:.4f} | "
              + " ".join(f"{t} {e[t]['f1']:.3f}" for t in ["PER", "ORG", "LOC", "MISC"]))

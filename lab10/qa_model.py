"""
LAB 10 - Extractive QA: tiền xử lý (character span → token span), hậu xử lý (logits → answer span),
EM / F1 theo chuẩn SQuAD, baseline keyword overlap và lớp QAModel để hỏi đáp trên tài liệu.

Chạy được cả trên máy (offline, dữ liệu ở ../hf_models) lẫn trên Google Colab (tự tải từ Hugging Face).
"""

import collections
import os
import re
import string
from pathlib import Path


def setup_env():
    """Trên máy: dùng ../hf_models và chạy offline. Trên Colab (không có thư mục đó): dùng cache mặc định, tải online."""
    local = Path(__file__).resolve().parent.parent / "hf_models"
    if local.is_dir():
        os.environ.setdefault("HF_HOME", str(local))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return local.is_dir()


IS_LOCAL = setup_env()          # phải chạy trước khi import transformers

import numpy as np

MODEL_NAME = "distilbert/distilbert-base-cased"


# ---------------------------------------------------------------- dữ liệu
def load_squad(split):
    """SQuAD v1.1 → DataFrame: id, title, context, question, answers (list text), starts (list char index)."""
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = snapshot_download("rajpurkar/squad", repo_type="dataset", allow_patterns=["plain_text/*.parquet"])
    df = pd.read_parquet(os.path.join(root, "plain_text", f"{split}-00000-of-00001.parquet"))
    df["starts"] = df.answers.map(lambda a: [int(x) for x in a["answer_start"]])
    df["answers"] = df.answers.map(lambda a: [str(x) for x in a["text"]])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- character span → token span
def prepare_train_features(tok, questions, contexts, starts, texts, max_length=384, stride=128):
    """Tokenize (question, context) với sliding window và gán start/end token cho từng window.

    Answer nằm trọn trong window: start/end = token chứa ký tự đầu / ký tự cuối của answer.
    Answer bị cắt hoặc nằm ngoài window: start = end = 0 (token [CLS]).
    """
    enc = tok(list(questions), list(contexts), truncation="only_second", max_length=max_length, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    feats = {"input_ids": [], "attention_mask": [], "start_positions": [], "end_positions": [], "sample_idx": []}
    for f, offsets in enumerate(enc["offset_mapping"]):
        s_idx = enc["overflow_to_sample_mapping"][f]
        a_start = starts[s_idx]
        a_end = a_start + len(texts[s_idx])                 # vị trí sau ký tự cuối (exclusive)
        seq = enc.sequence_ids(f)
        start_tok = end_tok = 0
        for t, (s, e) in enumerate(offsets):
            if seq[t] != 1:                                  # chỉ xét token của context
                continue
            if s <= a_start < e:
                start_tok = t
            if s < a_end <= e:
                end_tok = t
        if start_tok == 0 or end_tok == 0:                   # answer không nằm trọn trong window này
            start_tok = end_tok = 0
        feats["input_ids"].append(enc["input_ids"][f])
        feats["attention_mask"].append(enc["attention_mask"][f])
        feats["start_positions"].append(start_tok)
        feats["end_positions"].append(end_tok)
        feats["sample_idx"].append(s_idx)
    return feats


def prepare_eval_features(tok, questions, contexts, max_length=384, stride=128):
    """Như trên nhưng giữ offset_mapping (None cho token không thuộc context) để đổi ngược token → ký tự."""
    enc = tok(list(questions), list(contexts), truncation="only_second", max_length=max_length, stride=stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    offsets = [[o if enc.sequence_ids(f)[t] == 1 else None for t, o in enumerate(off)]
               for f, off in enumerate(enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "offset_mapping": offsets, "sample_idx": list(enc["overflow_to_sample_mapping"])}


# ---------------------------------------------------------------- logits → answer span
def postprocess(start_logits, end_logits, feats, contexts, n_best=20, max_answer_len=30):
    """Với mỗi câu hỏi, xét mọi window, chọn cặp (start, end) có start_logit + end_logit lớn nhất,
    với điều kiện: cả hai thuộc context, start ≤ end, độ dài ≤ max_answer_len token."""
    by_sample = collections.defaultdict(list)
    for f, s in enumerate(feats["sample_idx"]):
        by_sample[s].append(f)
    preds = []
    for s_idx in range(len(contexts)):
        best = {"text": "", "start_char": -1, "end_char": -1, "score": -1e9}
        for f in by_sample.get(s_idx, []):
            sl, el, off = start_logits[f], end_logits[f], feats["offset_mapping"][f]
            for i in np.argsort(sl[:len(off)])[-n_best:][::-1]:
                for j in np.argsort(el[:len(off)])[-n_best:][::-1]:
                    if off[i] is None or off[j] is None or j < i or j - i + 1 > max_answer_len:
                        continue
                    score = float(sl[i] + el[j])
                    if score > best["score"]:
                        a, b = off[i][0], off[j][1]
                        best = {"text": contexts[s_idx][a:b], "start_char": a, "end_char": b, "score": score}
        preds.append(best)
    return preds


# ---------------------------------------------------------------- EM / F1 (chuẩn SQuAD)
def normalize_answer(s):
    """Chữ thường, bỏ dấu câu, bỏ mạo từ a/an/the, gộp khoảng trắng."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score(pred, golds):
    """SQuAD có nhiều đáp án gold cho một câu hỏi: lấy điểm cao nhất."""
    return max(exact_match(pred, g) for g in golds), max(f1_score(pred, g) for g in golds)


def evaluate(preds, gold_lists):
    s = np.array([score(p, g) for p, g in zip(preds, gold_lists)])
    return {"EM": 100 * s[:, 0].mean(), "F1": 100 * s[:, 1].mean()}, s


# ---------------------------------------------------------------- baseline keyword overlap
STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did what which who whom
whose when where why how that this these those it its as into than then there their they he she his her
him we you your i me my not no can could would should will may might has have had about after before
during over under between through up down out if so such""".split())
WORD_RE = re.compile(r"\w+(?:[-'.]\w+)*")


def _content(words):
    return {w.lower() for w in words if w.lower() not in STOP}


def baseline_answer(question, context, max_len=5):
    """Question → câu có nhiều từ trùng nhất → trong câu đó, cụm từ liên tiếp không có trong câu hỏi,
    gần các từ trùng nhất. Câu hỏi when/how many/what year ưu tiên cụm có chữ số."""
    q_words = _content(WORD_RE.findall(question))
    want_num = bool(re.match(r"(?i)\s*(when|how (many|much|long|old)|what year|in what year)", question))
    sents = [(m.start(), m.group()) for m in re.finditer(r"[^.!?]+[.!?]*", context)]
    best_sent = max(sents, key=lambda s: len(q_words & _content(WORD_RE.findall(s[1]))), default=(0, context))
    s0, sent = best_sent
    words = [(m.group(), s0 + m.start(), s0 + m.end()) for m in WORD_RE.finditer(sent)]
    hit = [i for i, (w, _, _) in enumerate(words) if w.lower() in q_words]
    runs, cur = [], []
    for i, (w, _, _) in enumerate(words):
        if w.lower() in q_words:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    cands = []
    for r in runs:
        while r and words[r[0]][0].lower() in STOP:          # bỏ stopword ở hai đầu cụm
            r = r[1:]
        while r and words[r[-1]][0].lower() in STOP:
            r = r[:-1]
        if not r:
            continue
        r = r[:max_len]
        dist = min(abs(i - h) for i in r for h in hit) if hit else 99
        has_num = any(any(c.isdigit() for c in words[i][0]) for i in r)
        cands.append((-(has_num and want_num), dist, len(r), r))
    if not cands:
        return ""
    r = min(cands)[3]
    return context[words[r[0]][1]:words[r[-1]][2]]


# ---------------------------------------------------------------- hỏi đáp trên tài liệu
class QAModel:
    """Load model đã fine-tune và trả lời câu hỏi trên một context bất kỳ (tự chia window nếu dài)."""

    def __init__(self, model_dir, max_length=384, stride=128):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(self.device).eval()
        self.max_length, self.stride = max_length, stride

    def answer_batch(self, questions, contexts, batch_size=32):
        feats = prepare_eval_features(self.tok, questions, contexts, self.max_length, self.stride)
        S, E = [], []
        with self.torch.no_grad():
            for i in range(0, len(feats["input_ids"]), batch_size):
                ids = feats["input_ids"][i:i + batch_size]
                L = max(map(len, ids))
                x = self.torch.tensor([a + [self.tok.pad_token_id] * (L - len(a)) for a in ids], device=self.device)
                m = self.torch.tensor([[1] * len(a) + [0] * (L - len(a)) for a in ids], device=self.device)
                out = self.model(input_ids=x, attention_mask=m)
                S += list(out.start_logits.float().cpu().numpy())
                E += list(out.end_logits.float().cpu().numpy())
        return postprocess(S, E, feats, contexts)

    def answer(self, question, context):
        return self.answer_batch([question], [context])[0]


# ---------------------------------------------------------------- unit test
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    q, c = "What is the capital of France?", "The capital of France is Paris."

    def span(question, context, answer, start=None, **kw):
        start = context.index(answer) if start is None else start
        f = prepare_train_features(tok, [question], [context], [start], [answer], **kw)
        return [(s, e, tok.decode(f["input_ids"][k][s:e + 1])) for k, (s, e) in
                enumerate(zip(f["start_positions"], f["end_positions"]))]

    # 1. trường hợp chuẩn: "Paris" là token 14 (xem calculations.md)
    assert span(q, c, "Paris") == [(14, 14, "Paris")]
    assert span(q, c, "capital of France") == [(10, 12, "capital of France")]
    # 2. answer bắt đầu giữa token: "rance" nằm trong token "France" → lấy cả token
    assert span(q, c, "rance", start=c.index("France") + 1)[0][:2] == (12, 12)
    # 3. answer kết thúc giữa token: "Pari" → token "Paris"
    assert span(q, c, "Pari")[0][:2] == (14, 14)
    # 4. answer bị truncation: context dài hơn max_length, answer ở cuối → window đầu không thấy answer → [CLS]
    long_c = "Filler sentence number one. " * 60 + "The capital of France is Paris."
    f = prepare_train_features(tok, [q], [long_c], [long_c.index("Paris")], ["Paris"], max_length=128, stride=32)
    print("context dài → số window:", len(f["input_ids"]), "| (start, end) mỗi window:",
          list(zip(f["start_positions"], f["end_positions"])))
    assert (f["start_positions"][0], f["end_positions"][0]) == (0, 0)
    # 5. sliding window: các window không chứa answer nhận (0, 0), window cuối chứa đúng "Paris"
    assert all(p == (0, 0) for p in zip(f["start_positions"][:-1], f["end_positions"][:-1]))
    s, e = f["start_positions"][-1], f["end_positions"][-1]
    assert tok.decode(f["input_ids"][-1][s:e + 1]) == "Paris"

    # metric
    assert exact_match("the Paris", "Paris") == 1.0 and exact_match("the city of Paris", "Paris") == 0.0
    assert abs(f1_score("France", "Paris France") - 2 / 3) < 1e-9
    assert score("Denver", ["Denver Broncos", "Denver"]) == (1.0, 1.0)

    # hậu xử lý: logits giả, đỉnh ở token 14 → đọc lại đúng "Paris"
    ev = prepare_eval_features(tok, [q], [c])
    sl, el = np.zeros(len(ev["input_ids"][0])), np.zeros(len(ev["input_ids"][0]))
    sl[14] = el[14] = 5.0
    sl[3] = 9.0                                              # token "the" của câu hỏi: không được chọn
    assert postprocess([sl], [el], ev, [c])[0]["text"] == "Paris"

    print("baseline:", baseline_answer(q, c), "|", baseline_answer("When was the game played?",
          "The game was played on February 7, 2016, at Levi's Stadium."))
    print("All tests passed.")

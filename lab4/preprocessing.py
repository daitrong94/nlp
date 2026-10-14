"""
LAB 04 - Preprocessing cho Text Classification (IMDb sentiment).

Các hàm chính:
- load_imdb()            : đọc train/test IMDb (parquet), label 0 = negative, 1 = positive
- clean_text()           : bỏ thẻ HTML (<br />), lowercase, chuẩn hoá dấu nháy
- tokenize()             : regex tokenizer giống LAB 02/03
- inspect_dataset()      : số samples, số classes, missing text/label, duplicate, độ dài
- class_distribution()   : bảng Class / Count / Percentage
- deduplicate()          : bỏ document trùng trong cùng một tập
- remove_overlap()       : bỏ document của test đã xuất hiện trong train (tránh leakage)
- split_train_val()      : chia train/validation có stratify
- MeanEmbeddingVectorizer: Average Word2Vec, Word2Vec được train trong fit() nên chỉ thấy TRAIN

Quy ước: mọi thứ cần "học" từ dữ liệu (vocabulary, IDF, Word2Vec) chỉ được fit trên TRAIN,
validation/test chỉ transform.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0: "negative", 1: "positive"}

_HTML_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "imdb"


def load_imdb(data_dir=DEFAULT_DATA_DIR):
    """Đọc IMDb từ data_dir/imdb_{train,test}.parquet, nếu chưa có thì tải từ Hugging Face.

    Mặc định data_dir là thư mục imdb/ ở thư mục gốc (ngang hàng với lab4/),
    nên chạy script từ thư mục nào cũng được.
    """
    data_dir = Path(data_dir)
    frames = {}
    for split in ("train", "test"):
        path = data_dir / f"imdb_{split}.parquet"
        if not path.exists():
            from datasets import load_dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            load_dataset("stanfordnlp/imdb", split=split).to_pandas().to_parquet(path)
        frames[split] = pd.read_parquet(path)[["text", "label"]]
    return frames["train"], frames["test"]


def clean_text(text):
    """Bỏ thẻ HTML, lowercase, đổi ’ thành '."""
    text = _HTML_RE.sub(" ", text)
    return text.lower().replace("’", "'").replace("`", "'")


def tokenize(text):
    """Token gồm chữ/số, giữ dạng rút gọn (don't, isn't). Gọi clean_text trước."""
    return _TOKEN_RE.findall(text)


def preprocess(text):
    """clean_text + tokenize."""
    return tokenize(clean_text(text))


def inspect_dataset(df, name=""):
    """Thống kê cơ bản của một tập."""
    text = df["text"]
    n_tokens = text.fillna("").map(lambda t: len(preprocess(t)))
    return {
        "split": name,
        "samples": len(df),
        "classes": df["label"].nunique(),
        "missing_text": int(text.isna().sum() + (text.fillna("").str.strip() == "").sum()),
        "missing_label": int(df["label"].isna().sum()),
        "duplicate_docs": int(text.duplicated().sum()),
        "len_mean": n_tokens.mean(),
        "len_median": n_tokens.median(),
        "len_min": n_tokens.min(),
        "len_max": n_tokens.max(),
    }


def class_distribution(df):
    """Bảng Class / Count / Percentage."""
    counts = df["label"].map(LABEL_NAMES).value_counts().sort_index()
    return pd.DataFrame({"Count": counts, "Percentage": 100 * counts / counts.sum()}).rename_axis("Class")


def deduplicate(df):
    """Giữ lần xuất hiện đầu tiên của mỗi text."""
    return df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def remove_overlap(test_df, train_df):
    """Bỏ các document trong test có text trùng với train."""
    seen = set(train_df["text"])
    return test_df[~test_df["text"].isin(seen)].reset_index(drop=True)


def split_train_val(df, val_size=0.2, seed=42):
    """Chia train/validation, giữ nguyên tỉ lệ class."""
    tr, va = train_test_split(df, test_size=val_size, random_state=seed, stratify=df["label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Average Word2Vec: v(doc) = 1/n * sum_i v(w_i), chỉ tính các từ có trong vocabulary.

    fit() train Word2Vec (Skip-gram) trên các document được truyền vào, tức là TRAIN.
    Document không có từ nào trong vocabulary được biểu diễn bằng vector 0.
    """

    def __init__(self, vector_size=100, window=5, min_count=5, sg=1, epochs=10, workers=4, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):
        from gensim.models import Word2Vec
        sentences = [preprocess(t) for t in X]
        self.w2v_ = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, sg=self.sg, epochs=self.epochs,
                             workers=self.workers, seed=self.seed)
        return self

    def embed_tokens(self, tokens):
        wv = self.w2v_.wv
        vecs = [wv[w] for w in tokens if w in wv.key_to_index]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def transform(self, X):
        return np.vstack([self.embed_tokens(preprocess(t)) for t in X])


if __name__ == "__main__":
    # clean_text / tokenize
    assert clean_text("Great<br /><br />Movie’s END") == "great  movie's end"
    assert preprocess("This movie isn't GOOD!<br />10/10") == ["this", "movie", "isn't", "good", "10", "10"]

    # inspect / dedup / overlap
    toy = pd.DataFrame({"text": ["a good film", "bad", "a good film", "", "ok"], "label": [1, 0, 1, 0, 1]})
    st = inspect_dataset(toy)
    assert st["samples"] == 5 and st["classes"] == 2
    assert st["duplicate_docs"] == 1 and st["missing_text"] == 1
    assert len(deduplicate(toy)) == 4
    test = pd.DataFrame({"text": ["bad", "new review"], "label": [0, 1]})
    assert list(remove_overlap(test, toy)["text"]) == ["new review"]
    dist = class_distribution(toy)
    assert dist.loc["positive", "Count"] == 3 and abs(dist["Percentage"].sum() - 100) < 1e-9

    # split giữ tỉ lệ class
    big = pd.DataFrame({"text": [f"doc {i}" for i in range(100)], "label": [i % 2 for i in range(100)]})
    tr, va = split_train_val(big, val_size=0.2)
    assert len(va) == 20 and va["label"].sum() == 10 and not set(tr["text"]) & set(va["text"])

    # Average Word2Vec: trung bình đúng, bỏ qua thứ tự từ
    docs = ["this movie is good", "this movie is not good", "a bad movie"] * 50
    vec = MeanEmbeddingVectorizer(vector_size=10, min_count=1, epochs=5, workers=1).fit(docs)
    wv = vec.w2v_.wv
    expected = np.mean([wv[w] for w in ["this", "movie", "is", "good"]], axis=0)
    assert np.allclose(vec.transform(["this movie is good"])[0], expected, atol=1e-6)
    a, b = vec.transform(["good movie is this", "this movie is good"])
    assert np.allclose(a, b, atol=1e-6)
    assert np.allclose(vec.transform(["zzz qqq"])[0], 0)
    print("All tests passed.")

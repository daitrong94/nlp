"""
Part E — Core Implementation.

Tự triển khai TF-IDF và cosine similarity (không dùng TfidfVectorizer()
cho phần core). Corpus kiểm thử:

D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"
"""

import math
from collections import Counter

CORPUS = [
    "cat eats fish",
    "dog eats fish",
    "cat likes fish",
]


def tokenize(text):
    """Tách text thành list token (lowercase, split theo khoảng trắng)."""
    return text.lower().split()


def build_vocabulary(documents):
    """Trả về list các term duy nhất, sắp xếp (ví dụ theo alphabet)."""
    terms = set()
    for tokens in documents:
        terms.update(tokens)
    return sorted(terms)


def compute_counts(tokens, vocabulary):
    """Trả về dict {term: count} cho một document đã tokenize."""
    counter = Counter(tokens)
    return {term: counter.get(term, 0) for term in vocabulary}


def compute_tf(counts):
    """
    tf(t, d) = c(t, d) / sum_t' c(t', d)
    counts: dict {term: count} của một document.
    Trả về dict {term: tf}.
    """
    total = sum(counts.values())
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents, vocabulary):
    """
    idf(t) = log(N / df(t))
    documents: list các document đã tokenize (list[list[str]]).
    Trả về dict {term: idf}.
    """
    n = len(documents)
    doc_sets = [set(tokens) for tokens in documents]
    idf = {}
    for term in vocabulary:
        df = sum(1 for doc_set in doc_sets if term in doc_set)
        idf[term] = math.log(n / df) if df > 0 else 0.0
    return idf


def compute_tfidf(tf, idf):
    """tfidf(t, d) = tf(t, d) * idf(t). Trả về dict {term: tfidf}."""
    return {term: tf_value * idf.get(term, 0.0) for term, tf_value in tf.items()}


def cosine_similarity(vec_x, vec_y):
    """
    cos(x, y) = x . y / (||x||_2 * ||y||_2)
    vec_x, vec_y: dict {term: value} hoặc list số cùng chiều.
    """
    if isinstance(vec_x, dict) and isinstance(vec_y, dict):
        terms = set(vec_x) | set(vec_y)
        x = [vec_x.get(t, 0.0) for t in terms]
        y = [vec_y.get(t, 0.0) for t in terms]
    else:
        x, y = list(vec_x), list(vec_y)

    dot = sum(xi * yi for xi, yi in zip(x, y))
    norm_x = math.sqrt(sum(xi * xi for xi in x))
    norm_y = math.sqrt(sum(yi * yi for yi in y))
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)


if __name__ == "__main__":
    # --- Unit tests tối thiểu, mỗi hàm phải có ít nhất một test ---

    tokenized_docs = [tokenize(d) for d in CORPUS]
    vocab = build_vocabulary(tokenized_docs)
    print("Vocabulary:", vocab)

    counts_d1 = compute_counts(tokenized_docs[0], vocab)
    tf_d1 = compute_tf(counts_d1)
    assert abs(tf_d1["cat"] - 1 / 3) < 1e-9, "tf(cat, D1) phải bằng 1/3"

    idf = compute_idf(tokenized_docs, vocab)
    assert abs(idf["fish"] - 0.0) < 1e-9, "idf(fish) phải bằng 0 (df=N=3)"

    tfidf_d1 = compute_tfidf(tf_d1, idf)
    assert abs(tfidf_d1["fish"] - 0.0) < 1e-9, "tfidf(fish, D1) phải bằng 0"

    sim = cosine_similarity({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 0})
    assert 0 < sim < 1, "cosine similarity phải nằm trong (0, 1) cho ví dụ này"

    print("All tests passed.")

"""
LAB 03 - Core Implementation: Co-occurrence word embeddings.

Tự cài đặt pipeline  context -> co-occurrence -> vector -> similarity
(không dùng Word2Vec hay thư viện embedding).

Các hàm chính (mục 11 của đề):
- build_vocabulary()          : từ -> chỉ số hàng/cột của ma trận
- build_cooccurrence_matrix() : X ∈ R^{|V|x|V|}, rows = target words, columns = context words
- cosine_similarity()         : cos(u, v) = u·v / (|u| |v|)
- most_similar()              : top-k từ có cosine cao nhất với một từ

Mở rộng (mục 12 - từ co-occurrence matrix đến dense embedding):
- ppmi()             : Positive PMI, giảm ảnh hưởng của các từ quá phổ biến
- reduce_dimension() : truncated SVD, |V| chiều sparse -> k chiều dense
- analogy()          : b - a + c (ví dụ king - man + woman)

Quy ước context window:
- Window đối xứng k: với từ ở vị trí i, các từ ở vị trí i-k..i-1 và i+1..i+k
  (trong cùng một câu) là context.
- Từ ngoài vocabulary vẫn giữ vị trí trong câu nhưng không được đếm
  (ví dụ "the" bị loại khỏi vocabulary thì "the cat" không tạo cặp nào).
"""

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text):
    """Lowercase, giữ lại chữ/số (và dạng rút gọn như don't, it's)."""
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


def split_sentences(text):
    """Tách document thành các câu theo dấu . ! ? và xuống dòng."""
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def preprocess_documents(documents):
    """list[str] document -> list[list[str]] câu đã tokenize (bỏ câu rỗng)."""
    sentences = []
    for doc in documents:
        for sent in split_sentences(doc):
            tokens = tokenize(sent)
            if tokens:
                sentences.append(tokens)
    return sentences


def _as_token_lists(sentences):
    """Cho phép truyền câu dạng chuỗi hoặc list token."""
    return [tokenize(s) if isinstance(s, str) else list(s) for s in sentences]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(sentences, min_count=1, max_size=None, stopwords=None):
    """
    Trả về dict {word: index}, sắp theo tần suất giảm dần (bằng nhau thì theo
    thứ tự chữ cái). Index là số thứ tự hàng/cột trong ma trận co-occurrence.

    - min_count: bỏ các từ xuất hiện ít hơn min_count lần.
    - max_size : chỉ giữ max_size từ phổ biến nhất (None = không giới hạn).
    - stopwords: tập từ bị loại khỏi vocabulary (ví dụ {"the"}).
    """
    counts = Counter(w for sent in _as_token_lists(sentences) for w in sent)
    stopwords = set(stopwords or ())
    words = [w for w, c in counts.items() if c >= min_count and w not in stopwords]
    words.sort(key=lambda w: (-counts[w], w))
    if max_size is not None:
        words = words[:max_size]
    return {w: i for i, w in enumerate(words)}


def index_to_word(vocabulary):
    """dict {word: index} -> list word theo index."""
    words = [None] * len(vocabulary)
    for w, i in vocabulary.items():
        words[i] = w
    return words


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(sentences, vocabulary, window=2, weighting="count", dense=False):
    """
    Xây ma trận word-context X (|V| x |V|): X[w, c] = số lần c nằm trong
    window của w (cùng câu, khoảng cách 1..window, cả trái và phải).

    - weighting="count"   : mỗi cặp cộng 1 (như ví dụ trong đề).
    - weighting="harmonic": cặp cách nhau d từ cộng 1/d (context gần quan trọng hơn).
    - dense=False trả về scipy.sparse.csr_matrix, dense=True trả về numpy array.

    Ma trận đối xứng vì window đối xứng: X[w, c] = X[c, w].
    """
    if window < 1:
        raise ValueError("window phải >= 1")
    if weighting not in ("count", "harmonic"):
        raise ValueError("weighting phải là 'count' hoặc 'harmonic'")

    # Nối mọi câu thành một mảng id (-1 = ngoài vocabulary) kèm id câu,
    # rồi với mỗi khoảng cách d ghép vị trí i với i + d nếu cùng câu.
    ids, sent_ids = [], []
    for s, sent in enumerate(_as_token_lists(sentences)):
        ids.extend(vocabulary.get(w, -1) for w in sent)
        sent_ids.extend([s] * len(sent))
    ids = np.asarray(ids, dtype=np.int64)
    sent_ids = np.asarray(sent_ids, dtype=np.int64)

    rows, cols, vals = [], [], []
    for d in range(1, window + 1):
        left, right = ids[:-d], ids[d:]
        mask = (sent_ids[:-d] == sent_ids[d:]) & (left >= 0) & (right >= 0)
        a, b = left[mask], right[mask]
        weight = 1.0 if weighting == "count" else 1.0 / d
        # w ở bên trái c và c ở bên trái w đều là co-occurrence
        rows.extend([a, b])
        cols.extend([b, a])
        vals.append(np.full(2 * len(a), weight))

    n = len(vocabulary)
    if rows:
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    X = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    X.sum_duplicates()
    return X.toarray() if dense else X


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _to_dense_vector(v):
    if sp.issparse(v):
        return np.asarray(v.todense()).ravel()
    return np.asarray(v, dtype=np.float64).ravel()


def cosine_similarity(u, v):
    """cos(u, v) = u·v / (|u| |v|). Vector 0 thì trả về 0."""
    u, v = _to_dense_vector(u), _to_dense_vector(v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm == 0:
        return 0.0
    return float(np.dot(u, v) / norm)


def _row_norms(matrix):
    if sp.issparse(matrix):
        return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return np.linalg.norm(matrix, axis=1)


def similarity_to_all(vector, matrix):
    """Cosine giữa một vector và mọi hàng của matrix (trả về mảng |V|)."""
    vector = _to_dense_vector(vector)
    dots = np.asarray(matrix @ vector).ravel()
    norms = _row_norms(matrix) * np.linalg.norm(vector)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(norms > 0, dots / norms, 0.0)
    return sims


def word_similarity(w1, w2, matrix, vocabulary):
    """Cosine giữa hai từ; trả về None nếu một trong hai từ ngoài vocabulary."""
    if w1 not in vocabulary or w2 not in vocabulary:
        return None
    return cosine_similarity(matrix[vocabulary[w1]], matrix[vocabulary[w2]])


def most_similar_to_vector(vector, matrix, vocabulary, top_k=5, exclude=()):
    """Top-k từ (word, cosine) gần vector nhất, bỏ các từ trong exclude."""
    sims = similarity_to_all(vector, matrix)
    words = index_to_word(vocabulary)
    for w in exclude:
        if w in vocabulary:
            sims[vocabulary[w]] = -np.inf
    top_k = min(top_k, len(words) - len(set(exclude) & set(vocabulary)))
    best = np.argpartition(-sims, top_k - 1)[:top_k] if top_k > 0 else []
    best = sorted(best, key=lambda i: (-sims[i], words[i]))
    return [(words[i], float(sims[i])) for i in best]


def most_similar(word, matrix, vocabulary, top_k=5):
    """
    Top-k từ có cosine similarity cao nhất với `word` (không tính chính nó).
    Ví dụ: most_similar(word="doctor", matrix=X, vocabulary=vocab, top_k=5)
    """
    if word not in vocabulary:
        raise KeyError(f"'{word}' không có trong vocabulary")
    return most_similar_to_vector(matrix[vocabulary[word]], matrix, vocabulary,
                                  top_k=top_k, exclude=(word,))


def analogy(a, b, c, matrix, vocabulary, top_k=5):
    """
    a : b :: c : ?  ->  tìm từ gần b - a + c nhất (không tính a, b, c).
    Ví dụ: analogy("man", "king", "woman") ~ queen.
    Mỗi vector được chuẩn hoá về độ dài 1 trước khi cộng trừ (như gensim).
    """
    for w in (a, b, c):
        if w not in vocabulary:
            raise KeyError(f"'{w}' không có trong vocabulary")

    def unit(w):
        v = _to_dense_vector(matrix[vocabulary[w]])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    target = unit(b) - unit(a) + unit(c)
    return most_similar_to_vector(target, matrix, vocabulary, top_k=top_k, exclude=(a, b, c))


# ---------------------------------------------------------------------------
# Từ co-occurrence đến dense embedding
# ---------------------------------------------------------------------------

def ppmi(matrix, alpha=0.75):
    """
    Positive PMI: PPMI(w, c) = max(0, log P(w, c) / (P(w) P_alpha(c))).
    P_alpha(c) ∝ count(c)^alpha làm mượt phân phối context (alpha = 0.75 như
    negative sampling của Word2Vec), giảm việc PMI ưu ái context hiếm.
    Chỉ tính trên các ô khác 0 nên kết quả vẫn sparse.
    """
    X = sp.csr_matrix(matrix, dtype=np.float64)
    total = X.sum()
    if total == 0:
        return X.copy()
    row_sum = np.asarray(X.sum(axis=1)).ravel()
    col_sum = np.asarray(X.sum(axis=0)).ravel() ** alpha
    p_w = row_sum / total
    p_c = col_sum / col_sum.sum()

    coo = X.tocoo()
    pmi = np.log((coo.data / total) / (p_w[coo.row] * p_c[coo.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=X.shape)


def reduce_dimension(matrix, dim=100, power=0.5, seed=42):
    """
    Truncated SVD: X ≈ U_k S_k V_k^T. Embedding của từ = U_k S_k^power
    (|V| x dim, dense). power = 0.5 thường cho similarity tốt hơn power = 1.
    """
    X = sp.csr_matrix(matrix, dtype=np.float64)
    dim = min(dim, min(X.shape) - 1)
    v0 = np.random.default_rng(seed).uniform(-1, 1, min(X.shape))
    U, S, _ = svds(X, k=dim, v0=v0)
    order = np.argsort(-S)
    return U[:, order] * (S[order] ** power)


def matrix_stats(matrix):
    """Kích thước, số ô khác 0, density và bộ nhớ (dense vs sparse CSR)."""
    X = sp.csr_matrix(matrix)
    n_rows, n_cols = X.shape
    nnz = X.nnz
    return {
        "vocabulary_size": n_rows,
        "matrix_size": f"{n_rows} x {n_cols}",
        "cells": n_rows * n_cols,
        "non_zero": nnz,
        "density": nnz / (n_rows * n_cols) if n_rows * n_cols else 0.0,
        "dense_MB": n_rows * n_cols * 8 / 2**20,
        "sparse_MB": (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 2**20,
    }


# ---------------------------------------------------------------------------
# Unit test với corpus của Bài 1
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eps = 1e-9
    corpus = [
        "the cat eats fish",
        "the dog eats fish",
        "the cat likes milk",
        "the dog likes meat",
    ]

    # Vocabulary: 'the' bị loại như ví dụ trong đề
    vocab = build_vocabulary(corpus, stopwords={"the"})
    assert set(vocab) == {"cat", "dog", "eats", "likes", "fish", "milk", "meat"}
    assert index_to_word(vocab)[vocab["eats"]] == "eats"

    X = build_cooccurrence_matrix(corpus, vocab, window=1, dense=True)
    assert np.allclose(X, X.T)

    def row(w):
        return {c: X[vocab[w], vocab[c]] for c in vocab if X[vocab[w], vocab[c]]}

    # Kết quả tính tay trong calculations.md (Bài 1)
    assert row("cat") == {"eats": 1, "likes": 1}
    assert row("dog") == {"eats": 1, "likes": 1}
    assert row("eats") == {"cat": 1, "dog": 1, "fish": 2}
    assert row("likes") == {"cat": 1, "dog": 1, "milk": 1, "meat": 1}

    # Sparse và dense cho cùng kết quả
    Xs = build_cooccurrence_matrix(corpus, vocab, window=1)
    assert sp.issparse(Xs) and np.allclose(Xs.toarray(), X)

    # Window 2: "cat" thấy thêm "fish"/"milk", "the" vẫn không được đếm
    X2 = build_cooccurrence_matrix(corpus, vocab, window=2, dense=True)
    assert X2[vocab["cat"], vocab["fish"]] == 1 and X2[vocab["cat"], vocab["milk"]] == 1
    Xh = build_cooccurrence_matrix(corpus, vocab, window=2, weighting="harmonic", dense=True)
    assert abs(Xh[vocab["cat"], vocab["fish"]] - 0.5) < eps

    # Bài 2, 3: cosine
    assert abs(cosine_similarity([1, 2, 1], [2, 4, 2]) - 1.0) < eps
    assert abs(cosine_similarity([0.8, 0.1, 0.7], [0.7, 0.2, 0.8]) - 1.14 / np.sqrt(1.14 * 1.17)) < eps
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert abs(cosine_similarity(Xs[vocab["cat"]], Xs[vocab["dog"]]) - 1.0) < eps

    # most_similar: dog có context giống hệt cat
    top = most_similar(word="cat", matrix=Xs, vocabulary=vocab, top_k=2)
    assert top[0][0] == "dog" and abs(top[0][1] - 1.0) < eps and len(top) == 2
    assert all(w != "cat" for w, _ in most_similar("cat", X, vocab, top_k=10))
    assert word_similarity("cat", "unknown", Xs, vocab) is None

    # Analogy trên vector giả định của Bài 23 (mục 23 của đề)
    toy_vocab = {"king": 0, "man": 1, "woman": 2, "queen": 3, "apple": 4}
    toy = np.array([[8, 2, 7], [5, 1, 5], [5, 3, 5], [8, 4, 7], [0, 9, 0]], dtype=float)
    assert analogy("man", "king", "woman", toy, toy_vocab, top_k=1)[0][0] == "queen"

    # PPMI và SVD
    P = ppmi(Xs)
    assert P.shape == Xs.shape and (P.data > 0).all()
    E = reduce_dimension(P, dim=3)
    assert E.shape == (len(vocab), 3)
    assert abs(cosine_similarity(E[vocab["cat"]], E[vocab["dog"]]) - 1.0) < 1e-6

    stats = matrix_stats(Xs)
    assert stats["non_zero"] == 14 and stats["cells"] == 49

    print("All tests passed.")

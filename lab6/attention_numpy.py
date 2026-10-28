"""
LAB 06 - Scaled dot-product attention tự implement bằng NumPy.

    S = Q K^T / sqrt(d_k)        scores
    A = softmax(S)               attention weights, mỗi hàng có tổng = 1
    C = A V                      context

Các hàm:
- softmax(x, temperature)      : softmax ổn định số (trừ max trước khi lấy mũ)
- softmax_naive(x)             : softmax tính trực tiếp, chỉ để so sánh
- entropy(p)                   : entropy (nat) của phân phối attention
- scaled_dot_product_attention : trả về (scores, attention_weights, context)
- self_attention               : Q = X W_Q, K = X W_K, V = X W_V rồi gọi hàm trên

Q có shape (..., n_q, d_k), K có shape (..., n_k, d_k), V có shape (..., n_k, d_v).
mask (tuỳ chọn) có shape broadcast được tới (..., n_q, n_k), True = được attend, False = bỏ qua (ví dụ PAD).
"""

import numpy as np


def softmax(x, axis=-1, temperature=1.0):
    """softmax(x / T) ổn định số: e^(x_i - max) / sum_j e^(x_j - max)."""
    z = np.asarray(x, dtype=float) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_naive(x, axis=-1):
    """Tính trực tiếp e^x / sum e^x, tràn số với x lớn (để so sánh)."""
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(np.asarray(x, dtype=float))
        return e / np.sum(e, axis=axis, keepdims=True)


def entropy(p, axis=-1):
    """H(p) = -sum p log p (nat), quy ước 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True, temperature=1.0):
    """Trả về (scores, attention_weights, context).

    scores = Q K^T / sqrt(d_k) (hoặc không chia nếu scale=False),
    attention_weights = softmax(scores / temperature) theo trục key,
    context = attention_weights @ V.
    """
    Q, K, V = (np.asarray(M, dtype=float) for M in (Q, K, V))
    d_k = K.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)
    if scale:
        scores = scores / np.sqrt(d_k)
    masked = scores if mask is None else np.where(mask, scores, -np.inf)
    weights = softmax(masked, axis=-1, temperature=temperature)
    return scores, weights, weights @ V


def self_attention(X, W_Q, W_K, W_V, mask=None):
    """Self-attention: Q, K, V đều tạo từ cùng sequence X (shape (..., T, d_model))."""
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores, weights, context = scaled_dot_product_attention(Q, K, V, mask=mask)
    return {"Q": Q, "K": K, "V": V, "scores": scores, "weights": weights, "context": context}


# Ma trận giả định cho Bài self-attention (mục 21), xem calculations.md
SELF_ATTN_EXAMPLE = {
    "tokens": ["I", "like", "cats"],
    "X": np.array([[1, 0], [0, 1], [1, 1]], dtype=float),
    "W_Q": np.array([[1, 0], [1, 1]], dtype=float),
    "W_K": np.array([[1, 1], [0, 1]], dtype=float),
    "W_V": np.array([[1, 0], [0, 2]], dtype=float),
}


if __name__ == "__main__":
    # Bài 1: s = K q
    q = np.array([1, 2]); K = np.array([[1, 0], [0, 1], [1, 1]])
    assert np.array_equal(K @ q, [1, 2, 3])

    # Bài 2: softmax([1, 2, 3]), tổng = 1, thứ tự giữ nguyên
    a = softmax([1, 2, 3])
    assert np.allclose(a, [0.0900, 0.2447, 0.6652], atol=1e-4) and np.isclose(a.sum(), 1)
    assert np.allclose(softmax([2, 1, 0]), [0.6652, 0.2447, 0.0900], atol=1e-4)

    # Bài 3: c = 0.1 v1 + 0.2 v2 + 0.7 v3
    assert np.allclose(np.array([0.1, 0.2, 0.7]) @ np.array([[1, 0], [0, 1], [2, 2]]), [1.5, 1.6])

    # Bài 4: toàn bộ pipeline, không scale (đề dùng s = q^T k)
    s, w, c = scaled_dot_product_attention([[1, 0]], [[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 2], [2, 1]], scale=False)
    assert np.allclose(s, [[1, 0, 1]]) and np.allclose(w, [[0.4223, 0.1554, 0.4223]], atol=1e-4)
    assert np.allclose(c, [[1.2669, 0.7331]], atol=1e-4)

    # Mục 14: scaling với d_k = 4
    s, w, _ = scaled_dot_product_attention([[2, 0, 0, 0]], [[2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], np.eye(3))
    assert np.allclose(s, [[2, 1, 0]]) and np.allclose(w, softmax([2, 1, 0]))
    assert np.allclose(softmax([4, 2, 0]), [0.8668, 0.1173, 0.0159], atol=1e-4)

    # Mục 15: temperature thấp -> tập trung, cao -> phẳng
    ent = [entropy(softmax([2, 1, 0], temperature=T)) for T in (0.5, 1.0, 2.0)]
    assert ent[0] < ent[1] < ent[2] < np.log(3)
    assert np.isclose(entropy(softmax([1, 1, 1, 1])), np.log(4))

    # Mục 17: numerical stability
    assert np.all(np.isnan(softmax_naive([1000, 1001, 1002])))
    assert np.allclose(softmax([1000, 1001, 1002]), softmax([1, 2, 3]))
    assert np.allclose(softmax([1, 1, 10]), [1.234e-4, 1.234e-4, 0.99975], atol=1e-6)

    # Mục 21: self-attention ví dụ giả định
    ex = SELF_ATTN_EXAMPLE
    out = self_attention(ex["X"], ex["W_Q"], ex["W_K"], ex["W_V"])
    assert np.allclose(out["Q"] @ out["K"].T, [[1, 0, 1], [2, 1, 3], [3, 1, 4]])
    assert np.allclose(out["weights"], [[0.4011, 0.1978, 0.4011], [0.2840, 0.1400, 0.5760],
                                        [0.3057, 0.0743, 0.6200]], atol=1e-4)
    assert np.allclose(out["context"], [[0.8022, 1.1978], [0.8600, 1.4320], [0.9257, 1.3886]], atol=1e-4)
    assert np.allclose(out["weights"].sum(axis=1), 1)

    # Mask: key bị che nhận weight 0
    _, w, _ = scaled_dot_product_attention(np.ones((2, 3)), np.ones((4, 3)), np.eye(4),
                                           mask=np.array([True, True, False, False]))
    assert np.allclose(w, [[0.5, 0.5, 0, 0]] * 2)

    # Self-attention không có thông tin vị trí: hoán vị input thì output hoán vị theo
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4)); W = [rng.normal(size=(4, 4)) for _ in range(3)]
    perm = rng.permutation(5)
    assert np.allclose(self_attention(X[perm], *W)["context"], self_attention(X, *W)["context"][perm])

    # So với PyTorch (chỉ để kiểm chứng, không dùng trong core implementation)
    import torch
    import torch.nn.functional as F
    Qb, Kb, Vb = (rng.normal(size=(2, 6, 8)) for _ in range(3))
    mask = rng.random((2, 6, 6)) > 0.3
    mask[..., 0] = True                                   # mỗi hàng có ít nhất 1 key
    ref = F.scaled_dot_product_attention(*(torch.from_numpy(M) for M in (Qb, Kb, Vb)),
                                         attn_mask=torch.from_numpy(mask))
    assert np.allclose(scaled_dot_product_attention(Qb, Kb, Vb, mask=mask)[2], ref.numpy(), atol=1e-10)
    print("All tests passed.")

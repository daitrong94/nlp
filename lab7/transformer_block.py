"""
LAB 07 - Transformer Encoder block tối giản, tự ghép từ Linear / Softmax / LayerNorm / Dropout.

    x → Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm → output

(post-norm như sơ đồ trong đề và bài báo gốc). Không dùng nn.MultiheadAttention hay
nn.TransformerEncoderLayer; hai lớp đó chỉ dùng trong test để kiểm chứng.

Shape (B = batch, T = độ dài, d = d_model, h = n_heads, d_k = d / h):
    x                    (B, T, d)
    Q, K, V              (B, h, T, d_k)
    scores = QKᵀ/√d_k    (B, h, T, T)
    attention weights    (B, h, T, T)
    head outputs         (B, h, T, d_k) → concat (B, T, d) → W_O (B, T, d)
    FFN                  (B, T, d_ff) → (B, T, d)
    output               (B, T, d)
"""

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.head_mask = None                              # tensor (h,) 0/1 để tắt head khi phân tích
        self.shapes = {}                                   # ghi shape của lần forward gần nhất

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)        # (B, h, T, d_k)

    def forward(self, x, key_padding_mask=None):
        """key_padding_mask: (B, T), True = PAD (bị che). Trả về (output, attention weights)."""
        B, T, _ = x.shape
        Q, K, V = self.split_heads(self.W_Q(x)), self.split_heads(self.W_K(x)), self.split_heads(self.W_V(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)              # (B, h, T, T)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        heads = self.drop(weights) @ V                                      # (B, h, T, d_k)
        if self.head_mask is not None:
            heads = heads * self.head_mask.view(1, -1, 1, 1)
        concat = heads.transpose(1, 2).reshape(B, T, self.d_model)          # (B, T, d)
        out = self.W_O(concat)
        self.shapes = {"x": tuple(x.shape), "Q": tuple(Q.shape), "K": tuple(K.shape), "V": tuple(V.shape),
                       "scores": tuple(scores.shape), "weights": tuple(weights.shape),
                       "heads": tuple(heads.shape), "concat": tuple(concat.shape), "output": tuple(out.shape)}
        return out, weights


class FeedForward(nn.Module):
    """Position-wise FFN(x) = W2 ReLU(W1 x + b1) + b2, áp dụng độc lập cho từng vị trí."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(torch.relu(self.lin1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.shapes = {}

    def forward(self, x, key_padding_mask=None):
        a, weights = self.attn(x, key_padding_mask)
        x = self.norm1(x + self.drop1(a))                 # Add & Norm 1 (residual quanh attention)
        f = self.ffn(x)
        x = self.norm2(x + self.drop2(f))                 # Add & Norm 2 (residual quanh FFN)
        self.shapes = {**self.attn.shapes, "ffn_hidden": tuple(x.shape[:2]) + (self.ffn.lin1.out_features,),
                       "block_output": tuple(x.shape)}
        return x, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Shape theo ví dụ trong đề: B = 8, T = 100, d_model = 128, n_heads = 4 → d_k = 32
    blk = TransformerBlock(128, 4, 256)
    out, w = blk(torch.randn(8, 100, 128))
    s = blk.shapes
    assert s["Q"] == (8, 4, 100, 32) and s["scores"] == (8, 4, 100, 100) and out.shape == (8, 100, 128)
    for k, v in s.items():
        print(f"{k:13s} {v}")

    # 2. Multi-head attention khớp nn.MultiheadAttention khi chép cùng weight
    mha = MultiHeadSelfAttention(16, 4).eval()
    ref = nn.MultiheadAttention(16, 4, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.W_Q.weight, mha.W_K.weight, mha.W_V.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.W_Q.bias, mha.W_K.bias, mha.W_V.bias]))
        ref.out_proj.weight.copy_(mha.W_O.weight); ref.out_proj.bias.copy_(mha.W_O.bias)
    x = torch.randn(3, 7, 16)
    pad = torch.zeros(3, 7, dtype=torch.bool); pad[1, 5:] = True
    o1, w1 = mha(x, pad)
    o2, w2 = ref(x, x, x, key_padding_mask=pad, average_attn_weights=False)
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(w1, w2, atol=1e-6)

    # 3. Cả block khớp nn.TransformerEncoderLayer (post-norm, ReLU) khi chép cùng weight
    blk = TransformerBlock(16, 4, 32, dropout=0.0).eval()
    layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, norm_first=False).eval()
    with torch.no_grad():
        a = blk.attn
        layer.self_attn.in_proj_weight.copy_(torch.cat([a.W_Q.weight, a.W_K.weight, a.W_V.weight]))
        layer.self_attn.in_proj_bias.copy_(torch.cat([a.W_Q.bias, a.W_K.bias, a.W_V.bias]))
        layer.self_attn.out_proj.weight.copy_(a.W_O.weight); layer.self_attn.out_proj.bias.copy_(a.W_O.bias)
        for mine, theirs in [(blk.ffn.lin1, layer.linear1), (blk.ffn.lin2, layer.linear2),
                             (blk.norm1, layer.norm1), (blk.norm2, layer.norm2)]:
            theirs.weight.copy_(mine.weight); theirs.bias.copy_(mine.bias)
    assert torch.allclose(blk(x, pad)[0], layer(x, src_key_padding_mask=pad), atol=1e-5)

    # 4. Không có positional encoding: hoán vị input thì output hoán vị theo
    perm = torch.tensor([2, 0, 1, 4, 3, 6, 5])
    assert torch.allclose(blk(x[:, perm])[0], blk(x)[0][:, perm], atol=1e-5)

    # 5. Residual: block với attention và FFN bằng 0 vẫn giữ thông tin input (chỉ còn LayerNorm(x))
    zero = TransformerBlock(16, 4, 32, dropout=0.0).eval()
    with torch.no_grad():
        for p in list(zero.attn.parameters()) + list(zero.ffn.parameters()):
            p.zero_()
    ln = nn.LayerNorm(16)
    assert torch.allclose(zero(x)[0], ln(ln(x)), atol=1e-5)

    # 6. Số tham số không phụ thuộc số head khi d_model cố định
    n = [sum(p.numel() for p in MultiHeadSelfAttention(128, h).parameters()) for h in (1, 2, 4, 8)]
    assert len(set(n)) == 1 and n[0] == 4 * (128 * 128 + 128)
    print("All tests passed.")

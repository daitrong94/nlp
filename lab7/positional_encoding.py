"""
LAB 07 - Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

- sinusoidal_pe(max_len, d): ma trận NumPy (max_len, d)
- PositionalEncoding: module PyTorch, cộng PE vào embedding (X' = X + PE), không có tham số học
"""

import numpy as np
import torch
import torch.nn as nn


def sinusoidal_pe(max_len, d):
    """Ma trận PE kích thước (max_len, d). d phải chẵn."""
    assert d % 2 == 0
    pos = np.arange(max_len)[:, None]                 # (max_len, 1)
    i = np.arange(d // 2)[None, :]                    # (1, d/2)
    angle = pos / np.power(10000.0, 2 * i / d)        # (max_len, d/2)
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(angle)                       # chiều chẵn: sin
    pe[:, 1::2] = np.cos(angle)                       # chiều lẻ: cos
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.register_buffer("pe", torch.tensor(sinusoidal_pe(max_len, d_model), dtype=torch.float32))

    def forward(self, x):
        """x: (B, T, d_model) → x + PE[:T]."""
        return x + self.pe[: x.shape[1]][None]


if __name__ == "__main__":
    # Bài tính: d = 4, pos = 0, 1, 2
    pe = sinusoidal_pe(3, 4)
    expected = np.array([[0, 1, 0, 1],
                         [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)],
                         [np.sin(2), np.cos(2), np.sin(0.02), np.cos(0.02)]])
    assert np.allclose(pe, expected)
    print("PE (d = 4, pos = 0, 1, 2):\n", pe.round(4))

    # Mỗi vị trí có vector khác nhau, mọi giá trị trong [-1, 1]
    big = sinusoidal_pe(512, 128)
    assert len({tuple(r.round(6)) for r in big}) == 512 and np.abs(big).max() <= 1
    # ‖PE(pos)‖² = d/2 với mọi pos (mỗi cặp sin² + cos² = 1)
    assert np.allclose((big ** 2).sum(1), 64)

    # Module PyTorch cộng đúng PE
    m = PositionalEncoding(4, max_len=10)
    x = torch.zeros(2, 3, 4)
    assert torch.allclose(m(x)[0], torch.tensor(pe, dtype=torch.float32), atol=1e-6)
    print("All tests passed.")

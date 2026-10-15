"""
LAB 05 - Simple RNN cell tự implement bằng NumPy.

    h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

- SimpleRNNCell.forward(x_t, h_prev): một bước, x_t có shape (input_size,) hoặc (batch, input_size)
- SimpleRNNCell.forward_sequence(xs, h0): chạy lần lượt qua cả sequence, trả về mọi h_t
- from_torch()/to_torch(): chép weight sang/từ torch.nn.RNNCell để kiểm chứng

Quy ước giống PyTorch: W_xh có shape (hidden, input), W_hh có shape (hidden, hidden).
PyTorch có hai bias (b_ih, b_hh), ở đây gộp thành một b_h = b_ih + b_hh.
"""

import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # khởi tạo đều trong [-1/sqrt(H), 1/sqrt(H)], giống nn.RNNCell
        k = 1.0 / np.sqrt(hidden_size)
        rng = np.random.default_rng(seed)
        self.W_xh = rng.uniform(-k, k, (hidden_size, input_size))
        self.W_hh = rng.uniform(-k, k, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """Một timestep. Trả về (h_t, a_t), a_t là giá trị trước tanh."""
        x_t, h_prev = np.asarray(x_t, dtype=float), np.asarray(h_prev, dtype=float)
        # x @ W.T thay vì W @ x để chạy được cả với batch (batch, dim)
        a_t = x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h
        return np.tanh(a_t), a_t

    def forward_sequence(self, xs, h0=None):
        """xs: (T, input_size) hoặc (T, batch, input_size). Trả về mảng các h_t, shape (T, ..., hidden)."""
        h = np.zeros(xs.shape[1:-1] + (self.hidden_size,)) if h0 is None else np.asarray(h0, dtype=float)
        hs = []
        for x_t in xs:
            h, _ = self.forward(x_t, h)
            hs.append(h)
        return np.stack(hs)

    def num_parameters(self):
        return self.W_xh.size + self.W_hh.size + self.b_h.size

    @classmethod
    def from_torch(cls, torch_cell):
        cell = cls(torch_cell.input_size, torch_cell.hidden_size)
        cell.W_xh = torch_cell.weight_ih.detach().double().numpy().copy()
        cell.W_hh = torch_cell.weight_hh.detach().double().numpy().copy()
        cell.b_h = (torch_cell.bias_ih + torch_cell.bias_hh).detach().double().numpy().copy()
        return cell

    def to_torch(self):
        import torch
        cell = torch.nn.RNNCell(self.input_size, self.hidden_size, nonlinearity="tanh").double()
        with torch.no_grad():
            cell.weight_ih.copy_(torch.from_numpy(self.W_xh))
            cell.weight_hh.copy_(torch.from_numpy(self.W_hh))
            cell.bias_ih.copy_(torch.from_numpy(self.b_h))
            cell.bias_hh.zero_()
        return cell


def lab_example_cell():
    """Cell với đúng weight của Bài 1 trong đề."""
    cell = SimpleRNNCell(2, 2)
    cell.W_xh = np.array([[0.5, 0.2], [0.1, 0.4]])
    cell.W_hh = np.array([[0.3, 0.1], [0.2, 0.5]])
    cell.b_h = np.zeros(2)
    return cell


if __name__ == "__main__":
    # 1. Bài 1: a_t = [0.67, 0.30], h_t = tanh(a_t) = [0.585, 0.291]
    cell = lab_example_cell()
    h_t, a_t = cell.forward([1, 0], [0.5, 0.2])
    assert np.allclose(a_t, [0.67, 0.30])
    assert np.allclose(h_t, [0.58498, 0.29131], atol=1e-5)
    print("Bài 1: a_t =", a_t.round(4), " h_t =", h_t.round(5))

    import torch
    torch.manual_seed(0)

    # 2. So với torch.nn.RNNCell: chép weight từ torch sang cell tự viết
    tcell = torch.nn.RNNCell(4, 3).double()
    mine = SimpleRNNCell.from_torch(tcell)
    x = torch.randn(5, 4, dtype=torch.float64)       # batch 5
    h = torch.randn(5, 3, dtype=torch.float64)
    assert np.allclose(mine.forward(x.numpy(), h.numpy())[0], tcell(x, h).detach().numpy(), atol=1e-12)

    # 3. Cả sequence, so với torch.nn.RNN (dùng chung weight)
    rnn = torch.nn.RNN(4, 3, batch_first=False).double()
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(tcell.weight_ih); rnn.weight_hh_l0.copy_(tcell.weight_hh)
        rnn.bias_ih_l0.copy_(tcell.bias_ih); rnn.bias_hh_l0.copy_(tcell.bias_hh)
    xs = torch.randn(7, 2, 4, dtype=torch.float64)   # T = 7, batch 2
    out, h_last = rnn(xs)
    hs = mine.forward_sequence(xs.numpy())
    assert np.allclose(hs, out.detach().numpy(), atol=1e-12)
    assert np.allclose(hs[-1], h_last[0].detach().numpy(), atol=1e-12)

    # 4. Chiều ngược lại: to_torch() cho cùng kết quả với Bài 1
    t_h = cell.to_torch()(torch.tensor([[1.0, 0.0]], dtype=torch.float64),
                          torch.tensor([[0.5, 0.2]], dtype=torch.float64))
    assert np.allclose(t_h.detach().numpy()[0], h_t)

    # 5. Thứ tự có ý nghĩa: "dog bites man" ≠ "man bites dog", trung bình thì bằng nhau
    emb = {w: np.random.default_rng(i).normal(size=4) for i, w in enumerate(["dog", "bites", "man"])}
    s1 = np.stack([emb[w] for w in ["dog", "bites", "man"]])
    s2 = np.stack([emb[w] for w in ["man", "bites", "dog"]])
    assert np.allclose(s1.mean(0), s2.mean(0))
    assert not np.allclose(mine.forward_sequence(s1)[-1], mine.forward_sequence(s2)[-1])

    # 6. Số tham số: H*I + H*H + H (PyTorch có thêm một bias H nữa)
    assert mine.num_parameters() == 3 * 4 + 3 * 3 + 3
    assert sum(p.numel() for p in tcell.parameters()) == mine.num_parameters() + 3
    print("All tests passed.")

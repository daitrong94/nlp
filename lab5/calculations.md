# Calculations

> Xem bằng Markdown Preview (`Ctrl+Shift+V`) để hiển thị công thức.

## Bài 1 - Một bước RNN

$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ với $x_t = [1, 0]^\top$, $h_{t-1} = [0.5, 0.2]^\top$, $b_h = 0$:

$$
W_{xh}x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}, \qquad
W_{hh}h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}\begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}
$$

$$
a_t = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}, \qquad
h_t = \tanh(a_t) = \boxed{\begin{bmatrix} 0.585 \\ 0.291 \end{bmatrix}}
$$

$h_t$ vừa chứa token hiện tại ($0.5$) vừa mang thông tin cũ ($0.17$). Khớp với `rnn_cell.py` và `torch.nn.RNNCell`.

## Bài 2 - Information flow: "I did not like this movie"

- $h_1$ (I), $h_2$ (did): chưa có sentiment.
- $h_3$ (not): thêm tín hiệu phủ định.
- $h_4 = \tanh(W_{xh}x_{\text{like}} + W_{hh}h_3)$: "like" được đọc **cùng với** $h_3$ đã chứa "not". Vì tanh phi tuyến, model có thể học để "like" sau "not" thành tín hiệu tiêu cực.
- $h_6 = h_T$: tóm tắt cả câu, đưa vào classifier.

**Average Word2Vec:** $\vec{x} = \frac{1}{6}\sum \vec{v}_i$. "not" chỉ là một số hạng cộng thêm, không tác động lên "like", và mọi hoán vị cho cùng vector.

## Bài 3 - "dog bites man" và "man bites dog"

- **Average Word2Vec:** phép cộng giao hoán nên $\vec{x}_{S_1} = \vec{x}_{S_2}$, không phân biệt được.
- **RNN:** $h_3 = \tanh\big(W_{xh}v_{\text{man}} + W_{hh}\tanh(W_{xh}v_{\text{bites}} + W_{hh}\tanh(W_{xh}v_{\text{dog}}))\big)$. Token đứng đầu bị biến đổi qua nhiều lớp tanh hơn token đứng cuối, nên đổi thứ tự thì $h_3$ khác. Vị trí được mã hoá vào $h_T$.

## Mục 9 - Vanishing / exploding gradient

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \operatorname{diag}(1 - h_k^2)\, W_{hh}
$$

Tích có $T - t$ thừa số, mỗi thừa số có $1 - h_k^2 \le 1$:

- Độ lớn khoảng 0.9: $0.9^{200} \approx 7 \times 10^{-10}$, gradient không về tới token đầu (**vanishing**).
- Độ lớn khoảng 1.1: $1.1^{200} \approx 2 \times 10^{8}$ (**exploding**), xử lý bằng gradient clipping.

Sequence càng dài thì tích càng dài, nên dependency ở xa càng khó học.

## Mục 17 - LSTM gates

$$
c_t = f_t\,c_{t-1} + i_t\,\tilde{c}_t = 0.9 \times 0.8 + 0.2 \times 0.5 = 0.72 + 0.10 = \boxed{0.82}
$$

- $f_t \approx 1$: **giữ** thông tin cũ. $\partial c_t / \partial c_{t-1} = f_t \approx 1$ nên gradient gần như không bị co lại.
- $f_t \approx 0$: **xoá** thông tin cũ, bắt đầu lại từ token hiện tại.

## GRU

$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$: update gate $z_t$ gộp vai trò của forget gate và input gate, không có cell state riêng. GRU có 3 khối trọng số, LSTM có 4. Với input 100, hidden 128: RNN 29,440, GRU 88,320, LSTM 117,760 tham số (khớp PyTorch).

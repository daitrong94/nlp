# Calculations

> Công thức viết bằng LaTeX (`$...$`, `$$...$$`), xem bằng Markdown Preview của VS Code (`Ctrl+Shift+V`) hoặc trên GitHub.

## Bài 1 - Một bước RNN

$$
h_t = \tanh\left(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\right)
$$

Cho:

$$
x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}, \quad
W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}, \quad
W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}, \quad
b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

**Bước 1 - input-to-hidden:**

$$
W_{xh}\,x_t =
\begin{bmatrix} 0.5 \cdot 1 + 0.2 \cdot 0 \\ 0.1 \cdot 1 + 0.4 \cdot 0 \end{bmatrix} =
\begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}
$$

**Bước 2 - hidden-to-hidden:**

$$
W_{hh}\,h_{t-1} =
\begin{bmatrix} 0.3 \cdot 0.5 + 0.1 \cdot 0.2 \\ 0.2 \cdot 0.5 + 0.5 \cdot 0.2 \end{bmatrix} =
\begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} =
\begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}
$$

**Bước 3 - cộng lại (bias = 0):**

$$
a_t = W_{xh}\,x_t + W_{hh}\,h_{t-1} =
\begin{bmatrix} 0.5 + 0.17 \\ 0.1 + 0.20 \end{bmatrix} =
\begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}
$$

**Bước 4 - tanh**, với $\tanh(z) = \dfrac{e^{2z} - 1}{e^{2z} + 1}$:

$$
\tanh(0.67) = \frac{e^{1.34} - 1}{e^{1.34} + 1} \approx \frac{3.819 - 1}{3.819 + 1} = \frac{2.819}{4.819} \approx 0.585
$$

$$
\tanh(0.30) = \frac{e^{0.6} - 1}{e^{0.6} + 1} \approx \frac{1.822 - 1}{1.822 + 1} = \frac{0.822}{2.822} \approx 0.291
$$

$$
\boxed{h_t = \begin{bmatrix} 0.585 \\ 0.291 \end{bmatrix}}
$$

Kiểm chứng bằng code: `rnn_cell.py` cho $h_t = [0.58498,\ 0.29131]^\top$, khớp với `torch.nn.RNNCell` khi chép cùng weight.

Nhận xét: thành phần thứ nhất của $h_t$ nhận $0.5$ từ $x_t$ và $0.17$ từ $h_{t-1}$, tức $h_t$ vừa chứa token hiện tại vừa mang theo thông tin đã đọc trước đó. Hàm $\tanh$ giữ $h_t$ trong khoảng $(-1, 1)$.

## Bài 2 - Information flow: "I did not like this movie"

Model đọc từ trái sang phải, $h_0 = \mathbf{0}$:

| Bước | Token | Hidden state chứa gì (trực giác) |
|---|---|---|
| 1 | I | $h_1 = \tanh(W_{xh}x_{\text{I}} + b)$: chỉ có thông tin của "I", chủ ngữ ngôi thứ nhất, chưa có sentiment |
| 2 | did | $h_2 = \tanh(W_{xh}x_{\text{did}} + W_{hh}h_1 + b)$: "I did", đang chờ một động từ |
| 3 | not | $h_3$: thêm tín hiệu phủ định, model có thể bật một số chiều kiểu "đang trong phạm vi phủ định" |
| 4 | like | $h_4 = \tanh(W_{xh}x_{\text{like}} + W_{hh}h_3 + b)$: $x_{\text{like}}$ mang tín hiệu tích cực, nhưng $h_3$ đã chứa "not" |
| 5 | this | $h_5$: token trung tính, chủ yếu chép lại $h_4$ qua $W_{hh}$ |
| 6 | movie | $h_6 = h_T$: tóm tắt cả câu, đưa vào classifier $z = W_{hy}h_T + b_y$ |

Khi đọc "like", "not" ảnh hưởng tới $h_4$ qua số hạng $W_{hh}h_3$:

$$
h_4 = \tanh\big(\underbrace{W_{xh}\,x_{\text{like}}}_{\text{"like"}} + \underbrace{W_{hh}\,h_3}_{\text{chứa "not"}} + b\big)
$$

Vì $\tanh$ phi tuyến, kết quả không phải là "like" cộng "not" một cách độc lập: cùng một $x_{\text{like}}$ nhưng $h_3$ khác (có hoặc không có "not") thì $h_4$ khác. Model có thể học $W_{hh}$ sao cho "not" trong $h_3$ làm đổi dấu đóng góp của "like", tức là đọc "like" thành tín hiệu tiêu cực. Mô hình tuyến tính trên bag-of-words không có kiểu tương tác giữa hai từ như vậy.

**Average Word2Vec:**

$$
\vec{x} = \frac{1}{6}\left(\vec{v}_{\text{I}} + \vec{v}_{\text{did}} + \vec{v}_{\text{not}} + \vec{v}_{\text{like}} + \vec{v}_{\text{this}} + \vec{v}_{\text{movie}}\right)
$$

"not" chỉ là một trong sáu số hạng được cộng vào, không tác động lên "like". Mọi hoán vị của câu cho cùng một vector. Ở LAB 04, "this movie is good" và "this movie is not good" có $\cos = 0.976$, và Avg Word2Vec + LR vẫn đoán câu sau là positive.

Lưu ý: RNN chỉ *có khả năng* học tương tác này. Nó có học được hay không phụ thuộc vào dữ liệu và quá trình train. Khi "not" nằm cách "like" quá xa, tín hiệu có thể đã bị $W_{hh}$ và $\tanh$ làm mờ đi.

## Bài 3 - Sequence order: "dog bites man" và "man bites dog"

**Average Word2Vec:**

$$
\vec{x}_{S_1} = \tfrac{1}{3}\left(\vec{v}_{\text{dog}} + \vec{v}_{\text{bites}} + \vec{v}_{\text{man}}\right)
= \tfrac{1}{3}\left(\vec{v}_{\text{man}} + \vec{v}_{\text{bites}} + \vec{v}_{\text{dog}}\right) = \vec{x}_{S_2}
$$

Phép cộng có tính giao hoán nên hai câu có cùng vector, $\cos(\vec{x}_{S_1}, \vec{x}_{S_2}) = 1$. Vì vậy không classifier nào dựa trên vector này phân biệt được hai câu theo thứ tự. TF-IDF unigram cũng như vậy.

**RNN** ($h_0 = \mathbf{0}$, bỏ bias cho gọn):

| | $S_1$ = dog bites man | $S_2$ = man bites dog |
|---|---|---|
| $t=1$ | $h_1 = \tanh(W_{xh}v_{\text{dog}})$ | $h_1' = \tanh(W_{xh}v_{\text{man}})$ |
| $t=2$ | $h_2 = \tanh(W_{xh}v_{\text{bites}} + W_{hh}h_1)$ | $h_2' = \tanh(W_{xh}v_{\text{bites}} + W_{hh}h_1')$ |
| $t=3$ | $h_3 = \tanh(W_{xh}v_{\text{man}} + W_{hh}h_2)$ | $h_3' = \tanh(W_{xh}v_{\text{dog}} + W_{hh}h_2')$ |

Viết gọn cho $S_1$:

$$
h_3 = \tanh\Big(W_{xh}v_{\text{man}} + W_{hh}\tanh\big(W_{xh}v_{\text{bites}} + W_{hh}\tanh(W_{xh}v_{\text{dog}})\big)\Big)
$$

Ở $S_1$, $v_{\text{dog}}$ đi qua $W_{hh}$ hai lần và qua ba lớp $\tanh$, còn $v_{\text{man}}$ đi vào trực tiếp ở bước cuối. Ở $S_2$ thì ngược lại. Vì $\tanh$ phi tuyến và mỗi vị trí bị biến đổi khác nhau, nói chung $h_3 \neq h_3'$ (chỉ bằng nhau trong trường hợp suy biến như $W_{hh} = 0$). Vị trí của token quyết định nó bị biến đổi bao nhiêu lần, nên $h_T$ mã hoá được cả thứ tự. Một classifier trên $h_T$ vì vậy có thể phân biệt hai câu, nếu dữ liệu train có tín hiệu để học điều đó.

## Mục 9 - Vanishing / exploding gradient

Với $h_k = \tanh(W_{xh}x_k + W_{hh}h_{k-1} + b)$, đạo hàm qua một bước là:

$$
\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\!\left(1 - h_k^2\right) W_{hh}
$$

Qua nhiều bước, theo quy tắc chuỗi:

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^{T} \operatorname{diag}\!\left(1 - h_k^2\right) W_{hh}
$$

Tích này có $T - t$ thừa số. Mỗi thừa số có độ lớn cỡ $\left|1 - h_k^2\right| \cdot \lVert W_{hh} \rVert$, trong đó $1 - h_k^2 \le 1$ (bằng 1 chỉ khi $h_k = 0$ và nhỏ hơn nhiều khi $\tanh$ bão hoà).

- **Vanishing:** nếu mỗi thừa số có độ lớn khoảng $0.9$ thì

$$
0.9^{50} \approx 0.005, \qquad 0.9^{200} \approx 7 \times 10^{-10}
$$

  Gradient từ loss (tính trên $h_T$) gần như không về tới các token đầu, nên $W_{xh}$ và embedding không được cập nhật theo các token đó. Model vì vậy chỉ học được dependency gần cuối câu.

- **Exploding:** nếu mỗi thừa số có độ lớn khoảng $1.1$ thì $1.1^{200} \approx 1.9 \times 10^{8}$. Gradient bùng nổ, bước cập nhật quá lớn, loss dao động hoặc thành NaN. Cách xử lý thường dùng là gradient clipping: nếu $\lVert g \rVert > c$ thì thay $g$ bằng $c \cdot g / \lVert g \rVert$ (notebook dùng $c = 1.0$).

**Vì sao sequence càng dài càng khó học dependency ở xa:** số thừa số trong tích bằng khoảng cách $T - t$. Dependency càng xa thì tích càng dài và càng dễ tiến về $0$ (hoặc $\infty$). Ngoài ra, cùng một vector $h$ kích thước cố định phải chứa thông tin của mọi token đã đọc, nên thông tin cũ bị ghi đè dần.

## Mục 17 - LSTM gates

Các công thức của LSTM:

$$
\begin{aligned}
f_t &= \sigma\left(W_f[x_t, h_{t-1}] + b_f\right) && \text{forget gate} \\
i_t &= \sigma\left(W_i[x_t, h_{t-1}] + b_i\right) && \text{input gate} \\
\tilde{c}_t &= \tanh\left(W_c[x_t, h_{t-1}] + b_c\right) && \text{candidate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{cell state} \\
o_t &= \sigma\left(W_o[x_t, h_{t-1}] + b_o\right) && \text{output gate} \\
h_t &= o_t \odot \tanh(c_t) && \text{hidden state}
\end{aligned}
$$

Cho $f_t = 0.9$, $i_t = 0.2$, $\tilde{c}_t = 0.5$, $c_{t-1} = 0.8$:

$$
c_t = f_t \, c_{t-1} + i_t \, \tilde{c}_t = 0.9 \times 0.8 + 0.2 \times 0.5 = \underbrace{0.72}_{\text{giữ từ quá khứ}} + \underbrace{0.10}_{\text{thông tin mới}} = \boxed{0.82}
$$

Phần giữ lại từ quá khứ là $0.72$ (90% của $0.8$), phần thông tin mới chỉ là $0.10$. Cell state gần như được giữ nguyên và chỉ nhận thêm một chút từ token hiện tại.

- **Forget gate gần 1:** model giữ thông tin cũ. Khi đó $c_t \approx c_{t-1} + i_t\tilde{c}_t$ và

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1
$$

  nên cả thông tin lẫn gradient đi qua bước này gần như không bị suy giảm. Khác với RNN, ở đây không có phép nhân với $W_{hh}$ và $\tanh'$ ở mỗi bước. Đây là con đường giúp LSTM nhớ được dependency dài.

- **Forget gate gần 0:** model xoá thông tin cũ, $c_t \approx i_t\tilde{c}_t$, và cell state bắt đầu lại từ token hiện tại. Điều này hữu ích khi gặp ranh giới mới, ví dụ sau "but" hoặc khi review chuyển sang một đoạn khác.

Vì $f_t, i_t, o_t$ đều là sigmoid của $[x_t, h_{t-1}]$, model tự học khi nào giữ, khi nào ghi và khi nào xuất thông tin, tuỳ theo token và context. Gate là cơ chế điều khiển information flow, không phải một giá trị cố định.

## GRU (so sánh nhanh)

GRU có update gate $z_t$ và reset gate $r_t$, không có cell state riêng:

$$
\begin{aligned}
z_t &= \sigma\left(W_z[x_t, h_{t-1}] + b_z\right) \\
r_t &= \sigma\left(W_r[x_t, h_{t-1}] + b_r\right) \\
\tilde{h}_t &= \tanh\left(W_h[x_t,\ r_t \odot h_{t-1}] + b_h\right) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

Update gate làm cùng lúc việc của forget gate và input gate: $z_t$ nhỏ thì giữ $h_{t-1}$, $z_t$ lớn thì ghi đè bằng $\tilde{h}_t$. GRU có 3 khối trọng số so với 4 của LSTM, nên với cùng hidden size thì ít tham số hơn khoảng 25%:

$$
\#\text{params}_{\text{GRU}} = 3\left(H \cdot I + H \cdot H + 2H\right), \qquad
\#\text{params}_{\text{LSTM}} = 4\left(H \cdot I + H \cdot H + 2H\right)
$$

Với $I = 100$, $H = 128$ (cấu hình trong notebook): GRU có $88{,}320$ tham số, LSTM có $117{,}760$ và Simple RNN có $29{,}440$, khớp với số đếm của PyTorch.

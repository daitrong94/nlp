# Calculations

> Xem bằng Markdown Preview (`Ctrl+Shift+V`) để hiển thị công thức. Mọi kết quả được kiểm tra lại bằng `attention_numpy.py`.

$$
s_i = q^\top k_i, \qquad \alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad c = \sum_i \alpha_i v_i
$$

## Ví dụ trong đề

- $q = [1, 0]$, $k_1 = [1, 0]$, $k_2 = [0, 1]$: $s = [1, 0]$, nên query chú ý $k_1$ nhiều hơn.
- $\operatorname{softmax}([2, 1, 0]) = [0.665, 0.245, 0.090]$: token đầu nhận attention lớn nhất.

## Bài 1 - Scores

$q = [1, 2]$, các key là $[1, 0]$, $[0, 1]$, $[1, 1]$:

$$
s_1 = 1, \quad s_2 = 2, \quad s_3 = 1 + 2 = 3 \quad\Longrightarrow\quad \boxed{s = [1, 2, 3]}
$$

## Bài 2 - Softmax

$e^1 = 2.718$, $e^2 = 7.389$, $e^3 = 20.086$, tổng là $30.193$:

$$
\alpha = \boxed{[0.090,\ 0.245,\ 0.665]}, \qquad \sum_i \alpha_i = 1
$$

Score lớn nhất nhận weight lớn nhất vì $e^x$ đồng biến và mọi $\alpha_i$ có cùng mẫu số. Tỉ số $\alpha_i / \alpha_j = e^{s_i - s_j}$, nên softmax khuếch đại chênh lệch giữa các score.

## Bài 3 - Context vector

$$
c = 0.1\begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2\begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7\begin{bmatrix} 2 \\ 2 \end{bmatrix} = \boxed{\begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}}
$$

$v_3$ đóng góp nhiều nhất ($[1.4, 1.4]$) vì vừa có weight lớn nhất, vừa có độ lớn lớn nhất.

## Bài 4 - Toàn bộ attention

$q = [1, 0]$; keys $[1, 0]$, $[0, 1]$, $[1, 1]$; values $[1, 0]$, $[0, 2]$, $[2, 1]$.

1. Scores: $s = [1, 0, 1]$
2. Softmax: $[e, 1, e] / (2e + 1) = [0.422,\ 0.155,\ 0.422]$
3. Context: $0.422[1, 0] + 0.155[0, 2] + 0.422[2, 1] = \boxed{[1.267,\ 0.733]}$

## Scaling ($d_k = 4$)

| | Scores | Softmax |
|---|---|---|
| Không scale | $[4, 2, 0]$ | $[0.867,\ 0.117,\ 0.016]$ |
| Chia $\sqrt{4} = 2$ | $[2, 1, 0]$ | $[0.665,\ 0.245,\ 0.090]$ |

Scaling làm softmax **bớt tập trung**. Lý do cần chia: $\operatorname{Var}(q^\top k) = d_k$, nên khi $d_k$ lớn thì score lớn, softmax gần one-hot và gradient gần 0.

## Temperature

Với $s = [2, 1, 0]$:

| T | Softmax | Max weight | Entropy |
|---|---|---|---|
| 0.5 | $[0.867, 0.117, 0.016]$ | 0.867 | 0.44 |
| 1.0 | $[0.665, 0.245, 0.090]$ | 0.665 | 0.83 |
| 2.0 | $[0.506, 0.307, 0.186]$ | 0.506 | 1.02 |

T thấp thì phân phối tập trung, T cao thì phẳng (entropy tối đa $\ln 3 = 1.10$).

## Numerical stability

Với $x = [1000, 1001, 1002]$, tính $e^{1000}$ bị tràn số và cho nan. Trừ $\max(x) = 1002$ trước:

$$
\operatorname{softmax}([-2, -1, 0]) = [0.090,\ 0.245,\ 0.665]
$$

Phép trừ không đổi kết quả vì $e^{-m}$ là thừa số chung của tử và mẫu: $\dfrac{e^{x_i - m}}{\sum_j e^{x_j - m}} = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$.

## Self-attention: ["I", "like", "cats"]

Đề không cho số cụ thể, nên ma trận dưới đây là **giả định**:

$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad
W_Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

$$
Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 2 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 2 \end{bmatrix}, \quad
QK^\top = \begin{bmatrix} 1 & 0 & 1 \\ 2 & 1 & 3 \\ 3 & 1 & 4 \end{bmatrix}
$$

Chia $\sqrt{2}$ rồi lấy softmax theo từng hàng:

| | I | like | cats | Output $AV$ |
|---|---|---|---|---|
| I | 0.401 | 0.198 | 0.401 | $[0.802,\ 1.198]$ |
| like | 0.284 | 0.140 | 0.576 | $[0.860,\ 1.432]$ |
| cats | 0.306 | 0.074 | 0.620 | $[0.926,\ 1.389]$ |

Output của "like" lấy 57.6% từ "cats": mỗi token được biểu diễn lại bằng thông tin của các token khác, trong **một bước**.

## Kích thước attention matrix

| T | $T^2$ | Bộ nhớ (float32, 1 câu) |
|---|---|---|
| 10 | 100 | 400 B |
| 100 | 10,000 | 40 KB |
| 500 | 250,000 | 1 MB |
| 1,000 | 1,000,000 | 4 MB |
| 10,000 | 100,000,000 | 400 MB |

T gấp đôi thì attention matrix lớn gấp 4.

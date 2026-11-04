# Calculations

> Đề không cho số cụ thể của X, W ở Bài 1 và $W_2, b_2$ ở bài FFN, nên các ma trận đó là **giả định**. Mọi kết quả được kiểm lại trong notebook (mục 1).

## Bài 1 - Một attention head ($X \in \mathbb{R}^{3 \times 4}$, $d_k = 2$)

$$
X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}, \quad
W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \\ 0 & 1 \end{bmatrix}
$$

$$
Q = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
S = \frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 1.414 & 0 & 1.414 \\ 0 & 1.414 & 1.414 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}
$$

$$
A = \mathrm{softmax}(S) = \begin{bmatrix} 0.446 & 0.108 & 0.446 \\ 0.108 & 0.446 & 0.446 \\ 0.248 & 0.248 & 0.504 \end{bmatrix}, \qquad
Z = AV = \boxed{\begin{bmatrix} 0.892 & 0.554 \\ 0.554 & 0.554 \\ 0.752 & 0.497 \end{bmatrix}}
$$

Một head chính là attention của LAB 06, chỉ thêm ba ma trận chiếu $W_Q, W_K, W_V$ từ 4 chiều xuống $d_k = 2$.

## Bài 2 - Multi-head

$\text{head}_1, \text{head}_2 \in \mathbb{R}^{T \times 2}$, nên $\mathrm{Concat}(\text{head}_1, \text{head}_2) \in \mathbb{R}^{T \times 4}$, rồi nhân $W_O \in \mathbb{R}^{4 \times 4}$ được output $T \times 4$.

Vì sao concat trước khi nhân $W_O$:

- Mỗi head chỉ có $d_k = 2$ chiều. Concat ghép lại đúng $d_{model} = 4$ chiều, để cộng được với input ở residual $x + \mathrm{MHA}(x)$.
- $W_O$ trộn thông tin **giữa các head**. Nếu không có $W_O$, mỗi head chỉ ghi vào 2 chiều riêng của nó.

## Positional encoding ($d = 4$)

$PE(pos, 2i) = \sin(pos / 10000^{2i/d})$, $PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})$. Với $d = 4$: $i = 0$ chia cho 1, $i = 1$ chia cho $10000^{1/2} = 100$.

| pos | dim 0: $\sin(pos)$ | dim 1: $\cos(pos)$ | dim 2: $\sin(pos/100)$ | dim 3: $\cos(pos/100)$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |

Mỗi vị trí có một vector khác nhau. Hai chiều đầu đổi nhanh theo vị trí, hai chiều sau đổi chậm, giống kim giây và kim giờ.

## FFN

$x = [1, 2]$, $W_1 = I$, $b_1 = [0, -1]$:

$$
h = \mathrm{ReLU}(W_1 x + b_1) = \mathrm{ReLU}([1, 1]) = [1, 1]
$$

Giả định $W_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $b_2 = [0, 0.5]$:

$$
y = W_2 h + b_2 = [2,\ 0.5]
$$

FFN chỉ biến đổi **một vector tại một vị trí**, không nhìn sang token khác, nên nó không phải attention. ReLU tạo tính phi tuyến: với $x = [1, 0.5]$ thì $W_1x + b_1 = [1, -0.5]$, và ReLU cắt thành $[1, 0]$.

## Residual

$y = x + F(x)$. Với FFN ở trên: $y = [1, 2] + [2, 0.5] = [3, 2.5]$.

- Chỉ dùng $F(x)$: output phụ thuộc hoàn toàn vào $F$. Nếu $F$ học kém, thông tin của $x$ bị mất.
- Dùng $x + F(x)$: luôn giữ lại $x$, và $F$ chỉ cần học **phần thay đổi**. Gradient có đường đi thẳng qua phép cộng ($\partial y / \partial x = I + \partial F / \partial x$), nên dễ train hơn khi xếp nhiều lớp.

## Shape ($B = 8$, $T = 100$, $d_{model} = 128$, $h = 4$)

$d_k = 128 / 4 = 32$.

| Bước | Shape |
|---|---|
| X | 8 × 100 × 128 |
| Q, K, V | 8 × 4 × 100 × 32 |
| Attention scores, weights | 8 × 4 × 100 × 100 |
| Output mỗi head | 8 × 4 × 100 × 32 |
| Concat, $W_O$ | 8 × 100 × 128 |
| FFN hidden ($d_{ff} = 256$) | 8 × 100 × 256 |
| Output block | 8 × 100 × 128 |

Attention matrix có kích thước $T \times T$ vì mỗi token (hàng) tính score với mọi token (cột). Kiểm tra bằng `python transformer_block.py`.

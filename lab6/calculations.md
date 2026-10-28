# Calculations

> Công thức viết bằng LaTeX, xem bằng Markdown Preview của VS Code (`Ctrl+Shift+V`) hoặc trên GitHub. Mọi kết quả được kiểm lại bằng `attention_numpy.py` (`python attention_numpy.py`).

Ký hiệu chung:

$$
s_i = q^\top k_i, \qquad
\alpha_i = \operatorname{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad
c = \sum_i \alpha_i v_i
$$

## Mục 5 và 6 - Ví dụ trong đề

$q = [1, 0]^\top$, $k_1 = [1, 0]^\top$, $k_2 = [0, 1]^\top$: $s_1 = 1 \cdot 1 + 0 \cdot 0 = 1$ và $s_2 = 1 \cdot 0 + 0 \cdot 1 = 0$, nên query chú ý tới $k_1$ nhiều hơn.

$s = [2, 1, 0]$: $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$, tổng $= 11.107$.

$$
\alpha = \left[\frac{7.389}{11.107}, \frac{2.718}{11.107}, \frac{1}{11.107}\right] = [0.665,\ 0.245,\ 0.090]
$$

Token thứ nhất ($s = 2$) nhận attention lớn nhất.

## Bài 1 - Attention scores

$$
q = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \qquad
K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \quad (\text{mỗi hàng là một key } k_i)
$$

$$
\begin{aligned}
s_1 &= q^\top k_1 = 1 \cdot 1 + 2 \cdot 0 = 1 \\
s_2 &= q^\top k_2 = 1 \cdot 0 + 2 \cdot 1 = 2 \\
s_3 &= q^\top k_3 = 1 \cdot 1 + 2 \cdot 1 = 3
\end{aligned}
\qquad\Longrightarrow\qquad
\boxed{s = Kq = [1,\ 2,\ 3]}
$$

$k_3 = [1, 1]$ cùng hướng với $q$ nhiều nhất (cả hai thành phần đều dương), nên có score cao nhất.

## Bài 2 - Softmax

$s = [1, 2, 3]$:

$$
e^{1} = 2.7183, \quad e^{2} = 7.3891, \quad e^{3} = 20.0855, \qquad \sum_j e^{s_j} = 30.1929
$$

$$
\alpha = \left[\frac{2.7183}{30.1929},\ \frac{7.3891}{30.1929},\ \frac{20.0855}{30.1929}\right] = \boxed{[0.0900,\ 0.2447,\ 0.6652]}
$$

Tổng bằng 1:

$$
\sum_i \alpha_i = \frac{e^{1} + e^{2} + e^{3}}{e^{1} + e^{2} + e^{3}} = 1 \qquad (0.0900 + 0.2447 + 0.6652 = 0.9999 \approx 1 \text{ do làm tròn})
$$

Vì sao score lớn nhất nhận weight lớn nhất: mọi $\alpha_i$ có cùng mẫu số, còn tử số $e^{s_i}$ là hàm đồng biến, nên $s_i > s_j \Rightarrow \alpha_i > \alpha_j$. Hơn nữa

$$
\frac{\alpha_i}{\alpha_j} = e^{s_i - s_j}
$$

nên mỗi đơn vị chênh lệch score làm weight gấp $e \approx 2.72$ lần. Softmax khuếch đại chênh lệch: score 3 chỉ gấp 3 lần score 1 nhưng weight gấp $e^2 \approx 7.4$ lần.

## Bài 3 - Context vector

$\alpha = [0.1,\ 0.2,\ 0.7]$, $v_1 = [1, 0]^\top$, $v_2 = [0, 1]^\top$, $v_3 = [2, 2]^\top$:

$$
c = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix}
= \begin{bmatrix} 0.1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix}
= \boxed{\begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}}
$$

$v_3$ đóng góp nhiều nhất, $[1.4, 1.4]$, chiếm $1.4/1.5 = 93\%$ thành phần thứ nhất và $1.4/1.6 = 88\%$ thành phần thứ hai. Lý do là $v_3$ vừa có weight lớn nhất (0.7), vừa có độ lớn lớn nhất ($\lVert v_3 \rVert = 2\sqrt{2}$). Đóng góp của một token là $\alpha_i v_i$, phụ thuộc cả weight lẫn value, không chỉ weight.

## Bài 4 - Toàn bộ attention computation

$$
q = \begin{bmatrix} 1 \\ 0 \end{bmatrix};\quad
k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},\ k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix},\ k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix};\quad
v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},\ v_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix},\ v_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$

**Bước 1 - scores:**

$$
s_1 = 1 \cdot 1 + 0 \cdot 0 = 1, \qquad s_2 = 1 \cdot 0 + 0 \cdot 1 = 0, \qquad s_3 = 1 \cdot 1 + 0 \cdot 1 = 1 \qquad\Longrightarrow\qquad s = [1,\ 0,\ 1]
$$

**Bước 2 - softmax:** $e^1 = 2.7183$, $e^0 = 1$, tổng $= 2 \cdot 2.7183 + 1 = 6.4366$.

$$
\alpha = \left[\frac{2.7183}{6.4366},\ \frac{1}{6.4366},\ \frac{2.7183}{6.4366}\right] = [0.4223,\ 0.1554,\ 0.4223], \qquad \sum_i \alpha_i = 1
$$

**Bước 3 - weighted sum:**

$$
c = 0.4223 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.1554 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.4223 \begin{bmatrix} 2 \\ 1 \end{bmatrix}
= \begin{bmatrix} 0.4223 + 0 + 0.8446 \\ 0 + 0.3108 + 0.4223 \end{bmatrix}
= \boxed{\begin{bmatrix} 1.267 \\ 0.733 \end{bmatrix}}
$$

Nhận xét: $k_1$ và $k_3$ có cùng score với $q$ (vì $q$ chỉ "nhìn" thành phần thứ nhất), nên chia đều attention. $k_2$ vuông góc với $q$ nhưng vẫn nhận 15.5% weight: softmax không bao giờ cho weight bằng 0 tuyệt đối.

## Mục 14 - Scaling

$d_k = 4$, $QK^\top = [4, 2, 0]$, $\sqrt{d_k} = 2$:

$$
\frac{QK^\top}{\sqrt{4}} = [2,\ 1,\ 0]
$$

| | Scores | Softmax | Max weight |
|---|---|---|---|
| Không scaling | $[4, 2, 0]$ | $\left[\frac{54.598}{62.987}, \frac{7.389}{62.987}, \frac{1}{62.987}\right] = [0.867,\ 0.117,\ 0.016]$ | 0.867 |
| Scaling $/\sqrt{4}$ | $[2, 1, 0]$ | $[0.665,\ 0.245,\ 0.090]$ | 0.665 |

Scaling làm softmax **bớt tập trung** (phẳng hơn): chia score cho 2 làm chênh lệch giữa các score giảm một nửa, nên $\alpha_i/\alpha_j = e^{(s_i - s_j)/2}$ giảm. Token nhỏ nhất tăng từ 1.6% lên 9%.

Vì sao cần chia cho $\sqrt{d_k}$: nếu các thành phần của $q$ và $k$ độc lập, trung bình 0, phương sai 1, thì

$$
\operatorname{Var}(q^\top k) = \operatorname{Var}\Big(\sum_{m=1}^{d_k} q_m k_m\Big) = \sum_{m=1}^{d_k} \operatorname{Var}(q_m k_m) = d_k
$$

Độ lệch chuẩn của score tăng theo $\sqrt{d_k}$. Với $d_k = 512$, score dễ chênh nhau vài chục đơn vị, softmax bão hoà gần one-hot và gradient gần 0. Chia cho $\sqrt{d_k}$ đưa phương sai về 1 bất kể $d_k$.

## Mục 15 - Temperature

Với cùng $s = [2, 1, 0]$, $\alpha = \operatorname{softmax}(s/T)$, entropy $H(\alpha) = -\sum_i \alpha_i \ln \alpha_i$ (tối đa $\ln 3 = 1.099$ khi phân phối đều):

| T | $s/T$ | $\alpha$ | Max weight | Entropy | Nhận xét |
|---|---|---|---|---|---|
| 0.5 | $[4, 2, 0]$ | $[0.867,\ 0.117,\ 0.016]$ | 0.867 | 0.441 | Tập trung, gần chọn một token |
| 1.0 | $[2, 1, 0]$ | $[0.665,\ 0.245,\ 0.090]$ | 0.665 | 0.832 | Softmax gốc |
| 2.0 | $[1, 0.5, 0]$ | $[0.506,\ 0.307,\ 0.186]$ | 0.506 | 1.020 | Phẳng, gần đều |

Ví dụ entropy với $T = 0.5$: $-(0.867 \ln 0.867 + 0.117 \ln 0.117 + 0.016 \ln 0.016) = 0.124 + 0.251 + 0.066 = 0.441$.

Scaling ở mục 14 chính là temperature $T = \sqrt{d_k}$. Khi $T \to 0$, softmax tiến tới argmax (one-hot); khi $T \to \infty$, softmax tiến tới phân phối đều.

## Mục 17 - Numerical stability

Với $x = [1000, 1001, 1002]$, tính trực tiếp thì $e^{1000} \approx 10^{434}$, vượt giới hạn của float64 (khoảng $1.8 \times 10^{308}$). Kết quả là `inf / inf = nan`.

Trừ $m = \max(x) = 1002$:

$$
x - m = [-2,\ -1,\ 0] \quad\Longrightarrow\quad \operatorname{softmax}(x) = \frac{[e^{-2}, e^{-1}, e^{0}]}{e^{-2} + e^{-1} + 1} = \frac{[0.1353,\ 0.3679,\ 1]}{1.5032} = [0.0900,\ 0.2447,\ 0.6652]
$$

Phép trừ $\max(x)$ không làm đổi kết quả vì $e^{-m}$ là thừa số chung của tử và mẫu:

$$
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}} = \frac{e^{-m}\, e^{x_i}}{e^{-m} \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Sau khi trừ, số mũ lớn nhất là $e^0 = 1$, nên không bao giờ tràn số. Mẫu số luôn $\ge 1$, nên không bao giờ chia cho 0. Kết quả trùng với softmax của $[1, 2, 3]$ ở Bài 2, vì softmax chỉ phụ thuộc vào chênh lệch giữa các score.

## Mục 21 - Self-attention: ["I", "like", "cats"]

Đề không cho giá trị cụ thể của X và các ma trận W, nên mình tự chọn (giả định), với $d = d_k = 2$:

$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\begin{matrix} \leftarrow \text{I} \\ \leftarrow \text{like} \\ \leftarrow \text{cats} \end{matrix}
\qquad
W_Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

**Bước 1 - Q, K, V.** Hàng $i$ của $XW$ là embedding của token $i$ nhân với $W$:

$$
Q = XW_Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}, \qquad
K = XW_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 2 \end{bmatrix}, \qquad
V = XW_V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 2 \end{bmatrix}
$$

Ví dụ hàng "cats" của Q: $[1, 1] W_Q = [1 \cdot 1 + 1 \cdot 1,\ 1 \cdot 0 + 1 \cdot 1] = [2, 1]$.

**Bước 2 - $QK^\top$.** Phần tử $(i, j)$ là $q_i^\top k_j$, tức mức độ token $i$ "hỏi" về token $j$:

$$
QK^\top = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}
= \begin{bmatrix} 1 & 0 & 1 \\ 2 & 1 & 3 \\ 3 & 1 & 4 \end{bmatrix}
$$

Ví dụ hàng "like": $q_{\text{like}} = [1, 1]$, nên score với I, like, cats lần lượt là $1 + 1 = 2$, $0 + 1 = 1$ và $1 + 2 = 3$.

**Bước 3 - chia $\sqrt{d_k} = \sqrt{2} = 1.4142$:**

$$
S = \frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 0.7071 & 0 & 0.7071 \\ 1.4142 & 0.7071 & 2.1213 \\ 2.1213 & 0.7071 & 2.8284 \end{bmatrix}
$$

**Bước 4 - softmax theo từng hàng** (trừ max của hàng trước khi lấy mũ):

| Hàng | $S - \max$ | $e^{(\cdot)}$ | Tổng | $A$ (attention weights) |
|---|---|---|---|---|
| I | $[0, -0.7071, 0]$ | $[1, 0.4931, 1]$ | 2.4931 | $[0.4011,\ 0.1978,\ 0.4011]$ |
| like | $[-0.7071, -1.4142, 0]$ | $[0.4931, 0.2431, 1]$ | 1.7362 | $[0.2840,\ 0.1400,\ 0.5760]$ |
| cats | $[-0.7071, -2.1213, 0]$ | $[0.4931, 0.1199, 1]$ | 1.6130 | $[0.3057,\ 0.0743,\ 0.6200]$ |

Mỗi hàng có tổng bằng 1.

**Bước 5 - $AV$:**

$$
\text{Output} = AV = \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.2840 & 0.1400 & 0.5760 \\ 0.3057 & 0.0743 & 0.6200 \end{bmatrix}
\begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 2 \end{bmatrix}
= \boxed{\begin{bmatrix} 0.8022 & 1.1978 \\ 0.8600 & 1.4320 \\ 0.9257 & 1.3886 \end{bmatrix}}
$$

Ví dụ hàng "like": $0.2840 [1, 0] + 0.1400 [0, 2] + 0.5760 [1, 2] = [0.284 + 0.576,\ 0.280 + 1.152] = [0.860, 1.432]$.

Nhận xét:

- Output của "like" không còn là embedding của riêng "like" ($[0, 1]$). 57.6% trong đó lấy từ value của "cats" và 28.4% từ "I". Mỗi token được biểu diễn lại bằng cách trộn thông tin từ các token khác trong cùng câu, và đây là contextual representation.
- Mọi token tương tác trực tiếp với mọi token khác trong **một bước**, không phải truyền qua chuỗi $h_1 \to h_2 \to h_3$ như RNN.
- Attention matrix không đối xứng: "like" chú ý tới "cats" 0.576, nhưng "cats" chỉ chú ý tới "like" 0.074. Lý do là $W_Q \neq W_K$, nên "hỏi" và "được hỏi" là hai vai trò khác nhau.
- Trong ví dụ này, "cats" nhận attention lớn ở mọi hàng vì $k_{\text{cats}} = [1, 2]$ có độ lớn lớn nhất. Dot product phụ thuộc cả hướng lẫn độ lớn của vector.

## Mục 25 và 29 - Kích thước attention matrix

Full self-attention với $T$ token có ma trận $A \in \mathbb{R}^{T \times T}$. Bộ nhớ tính cho một head, một câu, float32 (4 byte):

| T | Attention matrix | Số phần tử $T^2$ | Bộ nhớ float32 | So với T = 10 |
|---|---|---|---|---|
| 10 | $10 \times 10$ | 100 | 400 B | 1× |
| 100 | $100 \times 100$ | 10,000 | 40 KB | 100× |
| 500 | $500 \times 500$ | 250,000 | 1 MB | 2,500× |
| 1,000 | $1000 \times 1000$ | 1,000,000 | 4 MB | 10,000× |
| 10,000 | $10^4 \times 10^4$ | 100,000,000 | 400 MB | 1,000,000× |

T tăng gấp đôi thì số phần tử tăng gấp 4. Thời gian tính $QK^\top$ và $AV$ là $O(T^2 d_k)$. Với batch 64 câu và 8 head, T = 10,000 cần $64 \times 8 \times 400$ MB $\approx 200$ GB chỉ để lưu A. Đây là lý do full self-attention khó dùng cho sequence rất dài.

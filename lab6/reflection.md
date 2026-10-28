# Reflection

## So sánh LAB 05 và LAB 06

| Property | RNN / LSTM | Self-attention | Quan sát trong lab |
|---|---|---|---|
| Sequence order | ✓ | Chỉ khi có positional information | Không có vị trí thì các câu cùng bag-of-words cho P(pos) giống hệt nhau |
| Tính toán | Tuần tự | Song song | 31.7 s/epoch so với 46.7 s/epoch của LSTM |
| Long-range | Khó | Dễ hơn | LSTM dồn 30% độ nhạy vào 20 token cuối, attention phân bố đều |
| Cost theo T | Tuyến tính | $O(T^2)$ | Attention chậm hơn LSTM từ khoảng T = 2000 |
| Context | $h_T$ | $\sum_t \alpha_t v_t$ | |
| Test F1 (IMDb) | LSTM 0.827 | 0.842 (+ pos: 0.834) | TF-IDF vẫn đạt 0.890 |

**Attention giải quyết vấn đề gì của RNN?**

- **Đường đi ngắn:** mọi token nối trực tiếp tới context, không phải qua chuỗi $h_1 \to \cdots \to h_T$. Từ khoá ở đầu review vẫn được nhận ra.
- **Không nén vào một vector:** giữ lại mọi $h_t$ và chọn theo query.
- **Song song:** train nhanh hơn LSTM khoảng 1.4 lần.

**Attention tạo ra vấn đề mới nào?**

- **Chi phí $O(T^2)$:** T = 4000 với batch 64 đã cần 3.9 GB chỉ cho ma trận attention.
- **Mất thứ tự** nếu không có positional encoding.
- **Một từ mạnh lấn át** cả câu, và model tự tin quá mức với câu ngắn.
- **Dễ diễn giải sai:** heatmap trông có cấu trúc, nhưng không chứng minh model hiểu câu.

Kết luận: attention không thay thế RNN "miễn phí". Nó đổi vấn đề đường đi dài lấy chi phí bình phương.

## Attention visualization không phải là reasoning

- Hai model có accuracy gần nhau cho heatmap khác hẳn nhau trên cùng một câu.
- Nhiều ô nổi bật không có nghĩa ngôn ngữ (yesterday → the 0.97).
- Model không vị trí cho adopted → cat 0.91 bất kể thứ tự từ, nên đó không thể là quan hệ ngữ pháp.

Heatmap chỉ cho biết attention được phân phối thế nào.

## Sang LAB 07

Cần positional encoding để có thứ tự, multi-head để học nhiều kiểu quan hệ cùng lúc, và residual cùng layer norm để xếp nhiều lớp. Ghép lại thành Transformer.

## Individual learning check

- **Attention score tính thế nào?** $s_i = q^\top k_i$, hoặc chia thêm $\sqrt{d_k}$.
- **Tại sao dùng softmax?** Biến score thành trọng số dương có tổng bằng 1, giữ thứ tự, và khả vi.
- **Q, K, V có vai trò gì?** Q là "đang tìm gì", K là "mỗi token có gì để so khớp", V là "thông tin được lấy ra".
- **Context vector tính thế nào?** $c = \sum_i \alpha_i v_i$.
- **Tại sao chia $\sqrt{d_k}$?** $\operatorname{Var}(q^\top k) = d_k$. Không chia thì softmax gần one-hot và gradient gần 0.
- **Self-attention khác attention thường ở đâu?** Q, K, V đều tạo từ cùng một sequence, nên mỗi token tương tác với mọi token khác.
- **Sequence dài gấp đôi thì sao?** Attention matrix lớn gấp 4.
- **Attention có chứng minh model đang reasoning không?** Không. Attention chỉ là trọng số trộn các value.
- **Tại sao attention xử lý long-range tốt hơn?** Mọi cặp token tương tác trực tiếp trong một bước, không có tích gradient dài.

# Reflection

## So sánh LAB 04 → LAB 07

| Property | TF-IDF | Avg Word2Vec | LSTM | Attention | Transformer |
|---|---|---|---|---|---|
| Representation | Sparse | Dense | Dense | Dense | Dense |
| Word order | Limited | ✗ | ✓ | Cần vị trí | Cần PE |
| Context-dependent | ✗ | ✗ | ✓ | ✓ | ✓ |
| Direct token interaction | ✗ | ✗ | Limited | ✓ | ✓ |
| Parallelizable | ✓ | ✓ | Limited | ✓ | ✓ |
| Long-range | Limited | ✗ | Difficult | Improved | Improved |
| Cost theo T | ~ tuyến tính | ~ tuyến tính | ~ tuyến tính | $O(T^2)$ | $O(T^2)$ |
| F1 trên IMDb | **0.890** | 0.859 | 0.818 | 0.842 | 0.811 |

(Mỗi lab dùng max_length khác nhau: LAB 06 là 200, LAB 07 là 128.)

## Mỗi bước giải quyết limitation nào của bước trước?

| Bước | Giải quyết được | Limitation còn lại |
|---|---|---|
| TF-IDF | Biểu diễn văn bản bằng số, đánh trọng số từ quan trọng | Không hiểu nghĩa: đồng nghĩa là hai chiều độc lập |
| Word2Vec | Từ gần nghĩa có vector gần nhau | Một từ một vector, trung bình làm mất thứ tự |
| LSTM | Đọc theo thứ tự, ngữ cảnh ảnh hưởng tới từ | Tuần tự, chậm, quên thông tin ở xa |
| Attention | Mọi token tương tác trực tiếp | Không có vị trí, chi phí $O(T^2)$ |
| Transformer | Ghép attention + PE + FFN + residual thành block xếp chồng được | Vẫn $O(T^2)$, cần nhiều dữ liệu, train từ đầu trên 20K review chưa vượt TF-IDF |

## Transformer khác gì so với self-attention của LAB 06?

- **Multi-head:** 4 head học 4 pattern khác nhau (JS divergence 0.30 - 0.36). F1 tăng từ 0.788 (1 head) lên 0.811 (4 head) mà không thêm tham số.
- **Positional encoding:** không có PE thì "dog bites man" và "man bites dog" giống hệt nhau. Có PE thì khác.
- **FFN:** biến đổi phi tuyến tại từng vị trí.
- **Residual + LayerNorm:** giữ thông tin input và giúp train ổn định.

## Bài học

- Transformer **không tự động thắng**: trên IMDb với 20K review, LSTM và Transformer gần bằng nhau, còn TF-IDF vẫn tốt nhất.
- Transformer tính song song được nhưng **không chắc nhanh hơn**: trên CPU, nó chậm hơn LSTM 3 lần.
- Tăng max_length giúp F1 (0.695 → 0.841) nhưng chi phí tăng theo bình phương.

## Sang LAB 08

Transformer ở lab này học từ đầu trên 20K review. Câu hỏi tiếp theo: tại sao không dùng một Transformer đã được train sẵn trên lượng văn bản khổng lồ (BERT), rồi fine-tune cho task? Ở LAB 08, `[CLS]`, PE, self-attention, FFN, residual và LayerNorm đều đã quen thuộc.

## Individual learning check

- **Tại sao cần positional encoding?** Self-attention không biết thứ tự. Không có PE thì đổi thứ tự từ cho cùng kết quả.
- **Multi-head khác một head ở đâu?** Nhiều head chạy song song, mỗi head có $W_Q, W_K, W_V$ riêng nên học quan hệ khác nhau. Kết quả được concat rồi nhân $W_O$.
- **$d_{model} = 128$, 8 head thì $d_k$ bằng bao nhiêu?** $d_k = 128 / 8 = 16$.
- **Tại sao attention matrix là $T \times T$?** Mỗi token (hàng) tính score với mọi token (cột).
- **Residual có vai trò gì?** Giữ lại input ($y = x + F(x)$) và cho gradient đi thẳng, giúp train sâu.
- **FFN làm gì?** Biến đổi phi tuyến từng vị trí riêng lẻ, không trộn giữa các token.
- **Tại sao Transformer song song hơn RNN?** Mọi vị trí được tính cùng lúc bằng phép nhân ma trận, không phải chờ bước trước.
- **Sequence dài gấp đôi thì attention matrix thế nào?** Lớn gấp 4.
- **Vì sao attention cao không phải là "lý do" của prediction?** Attention chỉ là trọng số trộn thông tin. Trong lab, attention chỉ trùng với token quan trọng nhất (đo bằng occlusion) trong khoảng một nửa số review.

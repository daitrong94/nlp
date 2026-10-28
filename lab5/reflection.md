# Reflection

## So sánh với LAB 04

| Property | TF-IDF + LR | Avg W2V + LR | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| Representation | Sparse | Dense | Dense | Dense | Dense |
| Word order | Limited | ✗ | ✓ | ✓ | ✓ |
| Context-dependent state | ✗ | ✗ | ✓ | ✓ | ✓ |
| Long dependency | ✗ | ✗ | Limited | Improved | Improved |
| Test F1 (IMDb) | **0.890** | 0.859 | 0.751 | 0.827 | 0.854 |
| Train time | 3.6 s | 88.6 s | 243 s | 294 s | 776 s |

**RNN/LSTM/GRU giải quyết được gì?**

- **Thứ tự từ:** câu cùng bag-of-words nhưng khác thứ tự cho P(pos) khác nhau.
- **Tương tác giữa các từ:** thêm "not" làm P(pos) của LSTM giảm từ 0.679 xuống 0.462, trong khi Avg Word2Vec vẫn đoán positive.
- **Contrast ngắn:** "I expected the movie to be bad, but it was actually wonderful": GRU đúng (0.593), TF-IDF sai (0.350).

**Vẫn còn vấn đề gì?**

- Không vượt được TF-IDF, dù tốn hơn 50 lần tham số và hơn 200 lần thời gian.
- Phải truncate: 40% review bị cắt, tỉ lệ lỗi cao hơn rõ.
- Gradient tới token xa yếu dần: Simple RNN khoảng $10^{-10}$. LSTM tốt hơn nhưng vẫn bỏ qua từ khoá ở đầu review dài.
- Cả review dồn vào một vector $h_T$, và vector này nghiêng về phần cuối.
- Tính tuần tự, không song song được theo thời gian.

## Learning curves

Cả ba model có train loss giảm liên tục, còn validation loss giảm tới epoch 4 - 5 rồi tăng: model bắt đầu **overfit**. GRU overfit mạnh nhất (train loss 0.04 ở epoch 8). Notebook giữ checkpoint có validation loss thấp nhất.

## Từ RNN tới Attention

Muốn dùng $x_1$ ở bước $T$, RNN phải truyền qua $T - 1$ bước, nên tín hiệu yếu dần và mọi thứ phải đi qua một vector cố định. Attention cho phép nhìn trực tiếp mọi $h_t$: $c = \sum_t \alpha_t h_t$. Khi đó token ở vị trí 5 hay 195 đều có đường đi ngắn tới output. Đây là nội dung của LAB 06.

## Individual learning check

- **RNN khác Average Word2Vec?** Avg W2V cộng trung bình nên mất thứ tự. RNN đọc tuần tự, $h_t$ phụ thuộc vào các từ đứng trước.
- **Hidden state $h_t$ là gì?** Vector tóm tắt các token từ $x_1$ tới $x_t$. $h_T$ là biểu diễn của cả câu.
- **Vì sao RNN bị vanishing gradient?** Gradient là tích của $T - t$ Jacobian, mỗi thừa số nhỏ hơn 1 nên tích giảm theo hàm mũ.
- **Forget gate dùng để làm gì?** Quyết định giữ bao nhiêu cell state cũ: gần 1 là giữ, gần 0 là xoá.
- **Vì sao sequence dài làm training chậm hơn?** Phải tính tuần tự $T$ bước ở cả forward và backward, và lưu activation của mọi bước.
- **GRU khác LSTM?** GRU không có cell state riêng, chỉ có 2 gate, ít hơn khoảng 25% tham số.
- **Train loss giảm, validation loss tăng nghĩa là gì?** Overfit: model ghi nhớ train và trở nên quá tự tin trên dữ liệu mới.
- **Truncate ở 30 tokens thì mất gì?** Mất phần kết luận, đoạn sau "but", và điểm số tự chấm ở cuối review.

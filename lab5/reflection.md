# Reflection

## So sánh với LAB 04 (mục 25)

| Property | TF-IDF + LR | Avg Word2Vec + LR | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| Sparse representation | ✓ | | | | |
| Dense representation | | ✓ | ✓ | ✓ | ✓ |
| Word order | Limited (chỉ qua n-gram) | ✗ | ✓ | ✓ | ✓ |
| Context-dependent state | ✗ | ✗ | ✓ | ✓ | ✓ |
| Long dependency | ✗ | ✗ | Limited | Improved | Improved |
| Computational cost | Low | Low | Higher | Higher | Higher |
| One vector per word | ✓ | ✓ | Embedding | Embedding | Embedding |
| **Test F1 trên IMDb** | **0.890** (unigram), 0.907 (uni + bigram) | 0.859 | 0.751 | 0.827 | 0.854 |
| Train time (CPU) | 3.6 s | 88.6 s | 243 s | 294 s | 776 s |

Số liệu của TF-IDF uni + bigram và Avg Word2Vec lấy từ LAB 04 (cùng split). Các model sequence dùng max_length = 200.

### RNN/LSTM/GRU giải quyết được gì mà Average Word2Vec không giải quyết tốt?

1. **Thứ tự từ.** Average Word2Vec có $\vec{x} = \frac{1}{T}\sum_t \vec{w}_t$, nên mọi hoán vị cho cùng một vector. RNN có $h_t = f(x_t, h_{t-1})$, nên vị trí quyết định token bị biến đổi bao nhiêu lần. Trong notebook, 4 cặp câu cùng bag-of-words cho P(pos) giống hệt nhau với TF-IDF, nhưng khác nhau với cả ba sequence model.
2. **Tương tác giữa các từ (context-dependent state).** Ở LAB 04, Avg Word2Vec + LR vẫn đoán "this movie is not good" là positive (0.714). Ở LAB 05, LSTM giảm từ 0.679 xuống 0.462 và GRU từ 0.845 xuống 0.561 khi thêm "not". Với "I did not like this movie at all", occlusion cho "like" đẩy về **negative** ($\Delta = -0.032$): cùng một từ nhưng đóng góp ngược dấu tuỳ context. Weight của LR thì không bao giờ đổi dấu như vậy.
3. **Contrast ngắn.** "I expected the movie to be bad, but it was actually wonderful": GRU 0.593 (đúng), TF-IDF 0.350 (sai). "I thought the movie would be terrible, but it was surprisingly good": GRU 0.680, TF-IDF 0.495.
4. **Embedding học cho đúng task.** Embedding được train cùng classifier bằng loss sentiment, nên không bị vấn đề "good gần bad" của Word2Vec học từ co-occurrence (cos 0.709 ở LAB 04).

### RNN/LSTM/GRU vẫn còn vấn đề gì?

1. **Không vượt được baseline đơn giản.** F1 tốt nhất (GRU 0.854) vẫn thấp hơn TF-IDF + LR (0.890), trong khi tốn khoảng 50 lần tham số và hơn 200 lần thời gian train.
2. **Truncation.** Phải chọn max_length cố định. Với L = 200, 40% review bị cắt, mất 31% token. Tỉ lệ lỗi trên review bị cắt là 19.5%, so với 12.1% trên review không bị cắt.
3. **Long-range dependency.** Gradient $\lVert \partial z/\partial e_t \rVert$ của Simple RNN ở token đầu chỉ còn khoảng $10^{-10}$ lần so với token cuối. LSTM và GRU giữ đường gradient tốt hơn nhiều, nhưng ở Experiment 6, khi câu quyết định đặt ở đầu và phía sau có khoảng 100 token filler, separation của LSTM vẫn về 0.
4. **Bottleneck của một vector.** Cả review dài hàng trăm token phải nén vào một $h_T \in \mathbb{R}^{128}$, và $h_T$ nghiêng về những gì đọc sau cùng (review 3827: "avoid this one till near the end" bị đoán positive 0.985).
5. **Tính tuần tự.** $h_t$ phải chờ $h_{t-1}$, nên không song song hoá được theo thời gian. Thời gian và bộ nhớ tăng tuyến tính theo L (LSTM: 26.9 → 106.3 MB activation mỗi batch khi L từ 50 lên 200).
6. **Overfit nhanh.** Validation loss tăng lại sau epoch 4 - 5, GRU từ 0.345 lên 0.554, trong khi train loss về 0.04.

## Learning curves (mục 23)

- Cả ba model có cùng mẫu: train loss giảm liên tục, validation loss giảm tới epoch 4 - 5 rồi tăng. Đây là dấu hiệu overfitting: model bắt đầu ghi nhớ train (embedding 2 triệu tham số cho 20K review) thay vì học quy luật tổng quát.
- GRU overfit sớm và mạnh nhất (best epoch 4, train loss 0.04 ở epoch 8). RNN overfit nhẹ nhất vì chưa fit được train tốt (train loss vẫn 0.33).
- Validation accuracy của LSTM vẫn tăng nhẹ (0.826 → 0.848) trong khi validation loss tăng (0.416 → 0.457). Model không sai nhiều hơn, nhưng sai với độ tự tin cao hơn, và BCE phạt nặng các dự đoán sai tự tin. Vì vậy phải nhìn cả đường cong chứ không chỉ epoch cuối, và phải ghi rõ chọn checkpoint theo metric nào (ở đây là validation loss).

## Từ RNN tới Attention

Muốn dùng thông tin của $x_1$ khi dự đoán ở bước $T$, RNN phải truyền nó qua $T - 1$ phép biến đổi:

$$
x_1 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T
$$

Thực nghiệm ở lab này cho thấy hai hệ quả: tín hiệu (và gradient) bị co lại theo khoảng cách, và mọi thông tin phải đi qua một vector kích thước cố định. Câu hỏi tự nhiên là: thay vì chỉ dùng $h_T$, có thể để classifier (hoặc mỗi token) **nhìn trực tiếp** vào mọi $h_t$ và tự chọn token nào quan trọng không? Chẳng hạn

$$
\alpha_t = \frac{\exp(\text{score}(q, h_t))}{\sum_{s} \exp(\text{score}(q, h_s))}, \qquad c = \sum_t \alpha_t h_t
$$

Khi đó "excellent" ở vị trí 5 và ở vị trí 195 đều có đường đi dài 1 tới output, và review 23571 hay 3827 có thể được quyết định bởi đúng câu kết luận thay vì bởi những token đọc cuối cùng. Đây là động lực của Attention (LAB 06), và Self-Attention / Transformer (LAB 07) bỏ hẳn phần recurrent để tính song song mọi vị trí.

## Individual learning check

- **RNN khác Average Word2Vec ở điểm nào?** Average Word2Vec cộng trung bình các word vector, nên mất thứ tự và mỗi từ đóng góp độc lập. RNN đọc tuần tự, $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$, nên $h_T$ phụ thuộc vào thứ tự, và đóng góp của một từ phụ thuộc vào các từ đứng trước ("like" sau "did not" đẩy về negative).
- **Hidden state $h_t$ có ý nghĩa gì?** Là bản tóm tắt (vector kích thước cố định, ở đây 128 chiều) của mọi token từ $x_1$ tới $x_t$, theo cách có ích cho task. $h_T$ là biểu diễn của cả câu đưa vào classifier.
- **Tại sao RNN có thể gặp vanishing gradient?** Vì $\partial h_T/\partial h_t = \prod_{k=t+1}^{T}\operatorname{diag}(1-h_k^2)W_{hh}$ là tích của $T - t$ ma trận. Nếu mỗi thừa số có độ lớn nhỏ hơn 1 thì tích giảm theo hàm mũ. Trong notebook, gradient của RNN ở token đầu chỉ bằng khoảng $10^{-10}$ lần token cuối.
- **Forget gate của LSTM dùng để làm gì?** Quyết định giữ bao nhiêu phần của cell state cũ: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. $f_t \approx 1$ thì giữ, và gradient qua $c$ gần như không co lại ($\partial c_t/\partial c_{t-1} = f_t$). $f_t \approx 0$ thì xoá thông tin cũ. Ví dụ $f = 0.9$, $c_{t-1} = 0.8$, $i = 0.2$, $\tilde{c} = 0.5$ cho $c_t = 0.72 + 0.10 = 0.82$.
- **Tại sao tăng sequence length làm training chậm hơn?** RNN phải tính $T$ bước tuần tự ở forward và $T$ bước ở backward (BPTT), lại phải lưu activation của mọi bước. Đo được: LSTM 11.8 → 34.9 s/epoch và 26.9 → 106.3 MB/batch khi L từ 50 lên 200.
- **GRU khác LSTM ở điểm nào?** GRU không có cell state riêng, chỉ có hidden state, và dùng 2 gate (update, reset) thay vì 3 gate cộng candidate. Update gate gộp vai trò forget và input: $h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde{h}_t$. Nhờ vậy GRU ít hơn khoảng 25% tham số (88,320 so với 117,760). Trên IMDb, GRU còn tốt hơn LSTM (F1 0.854 so với 0.827), nhưng chậm hơn trên CPU này vì LSTM có kernel tối ưu.
- **Tại sao training loss giảm nhưng validation loss tăng?** Model bắt đầu overfit: tiếp tục giảm loss trên train bằng cách ghi nhớ chi tiết riêng của train (embedding từ hiếm, cụm từ cụ thể) và trở nên quá tự tin, nên trên dữ liệu mới các dự đoán sai bị phạt nặng hơn. Ở GRU: train loss 0.25 → 0.04, validation loss 0.34 → 0.55 từ epoch 4 tới 8.
- **Nếu một review bị truncate ở 30 tokens, thông tin gì có thể bị mất?** Gần như toàn bộ: 30 token chỉ giữ 12.8% token của train, và thường là phần mở đầu (giới thiệu, tóm tắt cốt truyện). Mất phần kết luận ("overall I thoroughly enjoyed this film", "7/10", "avoid it"), mất các đoạn đảo chiều sau "but" / "however", và mất các lần nhắc lại cảm xúc. Với L = 50, F1 của LSTM giảm từ 0.827 xuống 0.742.

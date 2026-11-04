# Prediction

Ghi trước khi chạy experiment. Kết quả đối chiếu ở mục cuối.

## Prediction 1 - Transformer xử lý token song song hơn RNN?

- Prediction: Có.
- Reason: Self-attention tính mọi vị trí cùng lúc bằng phép nhân ma trận. RNN phải chờ $h_{t-1}$ mới tính được $h_t$.
- Confidence: 90%.

## Prediction 2 - Bỏ positional encoding, Transformer còn biết thứ tự không?

- Prediction: Không.
- Reason: Self-attention, FFN và LayerNorm đều không phụ thuộc vị trí. Đổi thứ tự input thì output chỉ đổi thứ tự theo.
- Confidence: 95%.

## Prediction 3 - Cùng tập token, khác thứ tự, không có PE thì phân biệt được không?

- Prediction: Không. Sau mean pooling, hai câu có **cùng** representation.
- Reason: Như Prediction 2. Ở LAB 06, self-attention không vị trí đã cho P(pos) giống hệt nhau.
- Confidence: 95%.

## Prediction 4 - Hai head có nhất thiết học cùng pattern không?

- Prediction: Không. Các head khác nhau nhưng có thể có head gần giống nhau (redundant).
- Reason: Mỗi head có $W_Q, W_K, W_V$ riêng, khởi tạo ngẫu nhiên khác nhau. Không có gì ép các head phải khác nhau.
- Confidence: 75%.

## Prediction 5 - Tăng số head có chắc tăng accuracy không?

- Prediction: Không. Chênh lệch giữa 1, 2, 4, 8 head sẽ nhỏ (dưới 1 - 2 điểm F1).
- Reason: Với $d_{model}$ cố định, tăng head không tăng tham số, chỉ chia nhỏ $d_k$. Sentiment chủ yếu dựa vào từ khoá nên không cần nhiều kiểu quan hệ.
- Confidence: 70%.

## Prediction 6 - Transformer luôn nhanh hơn RNN ở mọi sequence length?

- Prediction: Không. Nhanh hơn khi sequence ngắn và vừa, chậm hơn khi sequence rất dài.
- Reason: Attention tốn $O(T^2)$, RNN tốn $O(T)$. Ở LAB 06, self-attention vượt LSTM từ khoảng T = 2000.
- Confidence: 80%.

## Đối chiếu sau experiment

| # | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Song song hơn RNN | Tính song song được, nhưng trên CPU Transformer chậm hơn LSTM 3 lần (122 so với 41 s/epoch) | Đúng một phần | "Song song hơn" không có nghĩa là "nhanh hơn": mỗi block có 4 phép chiếu, FFN, LayerNorm và ma trận T × T, CPU không tận dụng hết được |
| 2 | Bỏ PE thì không biết thứ tự | Representation của hai câu đảo thứ tự chỉ khác 1e-6 | Đúng | |
| 3 | Cùng tập token thì không phân biệt được | Các cặp cùng bag-of-words cho P(pos) giống hệt nhau | Đúng | |
| 4 | Các head học pattern khác nhau | JS divergence giữa các head 0.30 - 0.36. Tắt head 4 không làm giảm F1 | Đúng | |
| 5 | Thêm head chênh dưới 1 - 2 điểm | 1 head 0.788, 4 head 0.811 (+2.3), 8 head 0.809 | Sai một phần | Từ 1 lên 4 head tăng nhiều hơn mình nghĩ. Đúng là không "càng nhiều càng tốt": 8 head không tăng thêm |
| 6 | Không luôn nhanh hơn RNN | Chậm hơn LSTM ngay ở T = 128 | Đúng | Còn chậm hơn dự đoán: không cần tới T = 2000 |

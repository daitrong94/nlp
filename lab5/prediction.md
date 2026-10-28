# Prediction

Prediction, Reason và Confidence ghi trước khi train. Dòng "Sau experiment" ghi sau khi có kết quả.

## Prediction 1 - TF-IDF + LR hay RNN xử lý word order trực tiếp hơn?

- Prediction: RNN. Nhưng chưa chắc RNN có F1 cao hơn.
- Reason: TF-IDF là bag-of-words, hoán vị từ cho cùng vector. RNN đọc tuần tự nên $h_T$ phụ thuộc thứ tự.
- Confidence: 90% (thứ tự), 50% (F1 cao hơn).
- Sau experiment: Đúng phần thứ tự: với các câu cùng bag-of-words, TF-IDF cho cùng P(pos), sequence model thì không. Nhưng F1 không vượt baseline: RNN 0.751, LSTM 0.827, GRU 0.854 so với TF-IDF 0.890.

## Prediction 2 - RNN hay LSTM xử lý dependency dài tốt hơn?

- Prediction: LSTM.
- Reason: RNN nhân nhiều lần với $W_{hh}$ và tanh' nên gradient co về 0. LSTM cập nhật cell state theo kiểu cộng, forget gate gần 1 giữ được gradient.
- Confidence: 80%.
- Sau experiment: Đúng. F1 0.827 so với 0.751. Gradient ở token đầu so với token cuối: RNN khoảng $10^{-10}$, LSTM khoảng 0.03. Nhưng LSTM vẫn bỏ qua từ khoá đặt ở đầu khi phía sau có khoảng 100 token.

## Prediction 3 - Sequence length 50 → 200, training RNN thay đổi thế nào?

- Prediction: Thời gian và bộ nhớ tăng khoảng 4 lần. Accuracy của Simple RNN có thể không tăng vì vanishing gradient.
- Reason: RNN tính tuần tự $T$ bước, BPTT cũng đi qua $T$ bước.
- Confidence: 75% (thời gian), 55% (accuracy).
- Sau experiment: Đúng về thời gian (×3.3) và bộ nhớ (×3.9). Sai về accuracy: F1 của RNN vẫn tăng (0.668 → 0.751), vì L lớn giữ được phần kết luận ở cuối review chứ không phải vì RNN nhớ xa hơn.

## Prediction 4 - Truncate về 30 tokens có ảnh hưởng không?

- Prediction: Có, accuracy giảm vài đến khoảng 10 điểm.
- Reason: Median review là 174 token. 30 token đầu thường chỉ là phần giới thiệu, kết luận nằm ở cuối.
- Confidence: 80%.
- Sau experiment: Đúng. Với L = 50, F1 của LSTM giảm từ 0.827 xuống 0.742. L = 30 chỉ giữ 12.8% token.

## Prediction 5 - LSTM có chắc tốt hơn RNN trên mọi dataset không?

- Prediction: Không.
- Reason: Với câu ngắn, dependency dài không quan trọng. Kết quả còn phụ thuộc hyperparameter và lượng dữ liệu.
- Confidence: 85%.
- Sau experiment: Đúng. Trên IMDb, LSTM hơn RNN 7.6 điểm F1, chủ yếu ở review dài. GRU đơn giản hơn LSTM nhưng lại tốt hơn (0.854).

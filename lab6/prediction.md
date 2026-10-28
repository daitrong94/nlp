# Prediction

Prediction, Reason và Confidence ghi trước khi chạy code. Dòng "Sau experiment" ghi sau khi có kết quả.

## Prediction 1 - Key có dot product lớn hơn thì attention weight thế nào?

- Prediction: Lớn nhất, gấp $e^{\Delta s}$ lần key khác.
- Reason: Softmax đồng biến, $\alpha_i / \alpha_j = e^{s_i - s_j}$.
- Confidence: 95%.
- Sau experiment: Đúng. "good" (score 4.17) nhận 0.381, "slow" (3.18) nhận 0.141. Tỉ số 2.70 ≈ $e^{0.99}$.

## Prediction 2 - $s = [1, 1, 1, 1]$

- Prediction: Đều, mỗi token 0.25.
- Reason: Softmax chỉ phụ thuộc chênh lệch giữa các score.
- Confidence: 95%.
- Sau experiment: Đúng, $[0.25, 0.25, 0.25, 0.25]$.

## Prediction 3 - $s = [1, 1, 10]$

- Prediction: Gần như toàn bộ vào token thứ ba (khoảng 0.9998).
- Reason: $e^{-9}$ rất nhỏ.
- Confidence: 90%.
- Sau experiment: Đúng, $[0.0001, 0.0001, 0.9998]$.

## Prediction 4 - 10 → 1000 tokens, chi phí thay đổi thế nào?

- Prediction: Số phần tử tăng 10,000 lần. Thời gian đo được tăng ít hơn vì chi phí cố định khi T nhỏ.
- Reason: Self-attention tính score cho $T^2$ cặp token.
- Confidence: 80%.
- Sau experiment: Đúng. Phần tử tăng 10,000 lần, thời gian NumPy tăng 745 lần. Bất ngờ: với T ≤ 1000, self-attention còn nhanh hơn LSTM vì tính song song được.

## Prediction 5 - Attention có luôn chọn đúng token quan trọng nhất?

- Prediction: Không.
- Reason: Attention được học để giảm loss, không phải để giải thích cho con người.
- Confidence: 80%.
- Sau experiment: Đúng. Token có attention cao nhất chỉ trùng với token quan trọng nhất (đo bằng occlusion) trong khoảng một nửa số review. Có review mà model dựa vào từ vô nghĩa như "prospective".

## Thêm - Self-attention vs LSTM trên IMDb

- Prediction: Gần bằng nhau. Self-attention không có positional encoding sẽ không phân biệt được thứ tự từ.
- Reason: Self-attention không có thông tin vị trí, nên hoán vị input thì output không đổi.
- Confidence: 65% (accuracy), 95% (thứ tự).
- Sau experiment: Đúng. F1: Self-Attention 0.842, LSTM 0.827. Các câu cùng bag-of-words cho P(pos) giống hệt nhau. Bất ngờ: model attention sai rất tự tin trên câu ngắn ("I like this movie." P = 0.000).

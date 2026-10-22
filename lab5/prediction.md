# Prediction

Prediction, Reason và Confidence được ghi trước khi train model nào. Dòng "Sau experiment" ghi sau khi có kết quả. Dataset: IMDb (sentiment, positive/negative), cùng split với LAB 04.

## Prediction 1 - TF-IDF + Logistic Regression hay RNN + Embedding xử lý word order trực tiếp hơn?

- Prediction: RNN + Embedding.
- Reason: TF-IDF unigram là bag-of-words, hoán vị các từ trong review cho ra cùng một vector, nên LR không thể phân biệt "not bad, good" với "not good, bad". RNN đọc từng token theo thứ tự, h_t = tanh(W_xh x_t + W_hh h_(t-1) + b), nên cùng một tập từ nhưng khác thứ tự sẽ cho h_T khác nhau. Tuy vậy "xử lý được thứ tự" không có nghĩa là accuracy cao hơn: TF-IDF + LR ở LAB 04 đã đạt F1 0.890.
- Confidence: 90% cho ý "trực tiếp hơn", 50% cho ý RNN có F1 cao hơn baseline.
- Sau experiment: Đúng phần chính, sai phần phụ. RNN + Embedding dùng thứ tự: với 4 cặp câu cùng bag-of-words (cos TF-IDF = 1), TF-IDF + LR cho P(pos) giống hệt nhau, còn RNN/LSTM/GRU cho kết quả khác nhau, thường theo hướng hợp lý (LSTM: "expected bad but it was great" 0.694 so với "expected great but it was bad" 0.328). Nhưng F1 thì không model sequence nào vượt TF-IDF + LR: 0.751 (RNN), 0.827 (LSTM), 0.854 (GRU) so với 0.890. Nguyên nhân chính là truncation ở 200 token và embedding học từ đầu.

## Prediction 2 - RNN hay LSTM xử lý dependency dài tốt hơn?

- Prediction: LSTM.
- Reason: Ở RNN, thông tin từ h_t muốn tới h_T phải đi qua T - t lần nhân với W_hh và đạo hàm tanh (≤ 1), nên gradient ∂h_T/∂h_t là tích của nhiều Jacobian và dễ tiến về 0. LSTM có cell state cập nhật theo kiểu cộng, c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t. Khi forget gate gần 1, đạo hàm ∂c_t/∂c_(t-1) ≈ f_t ≈ 1, thông tin và gradient đi qua nhiều bước mà không bị co lại nhiều.
- Confidence: 80%.
- Sau experiment: Đúng. Test F1 của LSTM là 0.827 so với 0.751 của RNN, cùng mọi điều kiện. Gradient của logit theo embedding ở token đầu (t = 1) so với token cuối (t = 200): RNN khoảng 10⁻¹⁰, LSTM khoảng 0.03, GRU khoảng 0.3. Accuracy trên review dài hơn 400 token: RNN 0.656, LSTM 0.732. Tuy vậy, ở Experiment 6 cả LSTM cũng không giữ được từ khoá đặt ở đầu khi phía sau có khoảng 100 token filler (separation 0.003). LSTM tốt hơn RNN nhưng chưa giải quyết được dependency dài.

## Prediction 3 - Sequence length tăng 50 → 200, training RNN thay đổi thế nào?

- Prediction: Thời gian train tăng gần tuyến tính theo độ dài (khoảng 4 lần) vì RNN phải tính tuần tự từng timestep và BPTT đi ngược qua cùng số bước. Bộ nhớ lưu activation cũng tăng khoảng 4 lần. Với Simple RNN, accuracy có thể không tăng, thậm chí giảm, vì hidden state cuối chủ yếu nhớ vài chục token gần cuối và gradient tới các token đầu bị vanishing. Với LSTM, accuracy sẽ tăng khi tăng độ dài.
- Confidence: 75% cho thời gian, 55% cho ý Simple RNN không cải thiện.
- Sau experiment: Đúng về thời gian, sai về accuracy của Simple RNN. Thời gian mỗi epoch của RNN 8.9 → 17.2 → 29.6 s (×3.3 khi L ×4), activation memory 2.9 → 11.3 MB (×3.9). Nhưng F1 của RNN vẫn tăng 0.668 → 0.700 → 0.751, không đứng yên như dự đoán. Lý do: tăng L không giúp RNN nhớ xa hơn (gradient ở t = 100 chỉ bằng 3·10⁻⁵ so với token cuối), mà làm nhiều review được giữ nguyên hơn (2.6% → 59.7%), nên phần kết luận ở cuối review lọt vào ngay trước h_T. LSTM giữ 100 token cuối (F1 0.838) còn tốt hơn LSTM giữ 200 token đầu (0.827).

## Prediction 4 - Truncate toàn bộ review về 30 tokens có ảnh hưởng không?

- Prediction: Có, ảnh hưởng rõ. Accuracy giảm vài điểm đến khoảng 10 điểm so với 200 tokens.
- Reason: Ở LAB 04, median độ dài review khoảng 174 token, nên 30 token đầu chỉ giữ chưa tới 20% nội dung. Phần đầu review thường là giới thiệu, tóm tắt cốt truyện. Câu kết luận ("9/10", "avoid it", "rent it today") và các từ đảo chiều ("but", "however") thường nằm ở cuối và sẽ bị mất.
- Confidence: 80%.
- Sau experiment: Đúng. Không thử đúng 30 token, nhưng với L = 50 (giữ 21% token, 2.6% review nguyên vẹn) F1 của LSTM giảm từ 0.827 xuống 0.742 (giảm 8.5 điểm), RNN từ 0.751 xuống 0.668. Với 30 token thì chỉ còn 12.8% token. Error analysis cũng cho thấy loại thông tin bị mất: các review sai vì truncation có kết luận nằm sau token 200 ("overall i would have to say that i thoroughly enjoyed this film").

## Prediction 5 - LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?

- Prediction: Không.
- Reason: Với câu ngắn (tweet, tiêu đề), dependency dài không quan trọng, RNN ít tham số hơn có thể bằng hoặc tốt hơn và train nhanh hơn. Kết quả còn phụ thuộc vào learning rate, khởi tạo, số epoch và lượng dữ liệu, LSTM nhiều tham số hơn có thể overfit trên dataset nhỏ. Trên IMDb với review dài thì dự đoán LSTM tốt hơn RNN rõ rệt.
- Confidence: 85%.
- Sau experiment: Đúng với dữ liệu hiện có, nhưng chỉ thử một dataset. Trên IMDb, LSTM hơn RNN 7.6 điểm F1. Khoảng cách nhỏ hơn ở review ngắn (≤ 50 token: accuracy 0.877 so với 0.832) và lớn ở review dài (> 400 token: 0.732 so với 0.656), phù hợp với lập luận "lợi thế của LSTM nằm ở dependency dài". GRU, đơn giản hơn LSTM, lại tốt hơn LSTM (F1 0.854), nên "phức tạp hơn" không đồng nghĩa với "tốt hơn". LSTM cũng tốn activation memory gấp khoảng 9 lần RNN.

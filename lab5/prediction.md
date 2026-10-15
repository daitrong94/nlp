# Prediction

Prediction, Reason và Confidence được ghi trước khi train model nào. Dòng "Sau experiment" ghi sau khi có kết quả. Dataset: IMDb (sentiment, positive/negative), cùng split với LAB 04.

## Prediction 1 - TF-IDF + Logistic Regression hay RNN + Embedding xử lý word order trực tiếp hơn?

- Prediction: RNN + Embedding.
- Reason: TF-IDF unigram là bag-of-words, hoán vị các từ trong review cho ra cùng một vector, nên LR không thể phân biệt "not bad, good" với "not good, bad". RNN đọc từng token theo thứ tự, h_t = tanh(W_xh x_t + W_hh h_(t-1) + b), nên cùng một tập từ nhưng khác thứ tự sẽ cho h_T khác nhau. Tuy vậy "xử lý được thứ tự" không có nghĩa là accuracy cao hơn: TF-IDF + LR ở LAB 04 đã đạt F1 0.890.
- Confidence: 90% cho ý "trực tiếp hơn", 50% cho ý RNN có F1 cao hơn baseline.

## Prediction 2 - RNN hay LSTM xử lý dependency dài tốt hơn?

- Prediction: LSTM.
- Reason: Ở RNN, thông tin từ h_t muốn tới h_T phải đi qua T - t lần nhân với W_hh và đạo hàm tanh (≤ 1), nên gradient ∂h_T/∂h_t là tích của nhiều Jacobian và dễ tiến về 0. LSTM có cell state cập nhật theo kiểu cộng, c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t. Khi forget gate gần 1, đạo hàm ∂c_t/∂c_(t-1) ≈ f_t ≈ 1, thông tin và gradient đi qua nhiều bước mà không bị co lại nhiều.
- Confidence: 80%.

## Prediction 3 - Sequence length tăng 50 → 200, training RNN thay đổi thế nào?

- Prediction: Thời gian train tăng gần tuyến tính theo độ dài (khoảng 4 lần) vì RNN phải tính tuần tự từng timestep và BPTT đi ngược qua cùng số bước. Bộ nhớ lưu activation cũng tăng khoảng 4 lần. Với Simple RNN, accuracy có thể không tăng, thậm chí giảm, vì hidden state cuối chủ yếu nhớ vài chục token gần cuối và gradient tới các token đầu bị vanishing. Với LSTM, accuracy sẽ tăng khi tăng độ dài.
- Confidence: 75% cho thời gian, 55% cho ý Simple RNN không cải thiện.

## Prediction 4 - Truncate toàn bộ review về 30 tokens có ảnh hưởng không?

- Prediction: Có, ảnh hưởng rõ. Accuracy giảm vài điểm đến khoảng 10 điểm so với 200 tokens.
- Reason: Ở LAB 04, median độ dài review khoảng 174 token, nên 30 token đầu chỉ giữ chưa tới 20% nội dung. Phần đầu review thường là giới thiệu, tóm tắt cốt truyện. Câu kết luận ("9/10", "avoid it", "rent it today") và các từ đảo chiều ("but", "however") thường nằm ở cuối và sẽ bị mất.
- Confidence: 80%.

## Prediction 5 - LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?

- Prediction: Không.
- Reason: Với câu ngắn (tweet, tiêu đề), dependency dài không quan trọng, RNN ít tham số hơn có thể bằng hoặc tốt hơn và train nhanh hơn. Kết quả còn phụ thuộc vào learning rate, khởi tạo, số epoch và lượng dữ liệu, LSTM nhiều tham số hơn có thể overfit trên dataset nhỏ. Trên IMDb với review dài thì dự đoán LSTM tốt hơn RNN rõ rệt.
- Confidence: 85%.

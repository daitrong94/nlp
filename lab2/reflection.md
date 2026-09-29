# Reflection

## Câu 1 - Nếu tăng n, mô hình nhận thêm thông tin gì?

Mô hình nhận thêm thông tin về thứ tự và ngữ cảnh cục bộ: n - 1 từ đứng trước thay vì ít hơn, nên nắm được collocation, cụm cố định và một phần ràng buộc ngữ pháp gần (ví dụ the fact → that). Trong thí nghiệm, training perplexity MLE giảm từ 1,434 (unigram) xuống 125 (bigram) và 9.9 (trigram).

## Câu 2 - Tại sao tăng n lại làm sparsity tăng?

Số n-gram có thể có tăng theo V^n trong khi lượng dữ liệu cố định, nên mỗi n-gram được quan sát ít lần hơn và phần lớn tổ hợp chưa bao giờ xuất hiện. Trong train: số n-gram unique tăng 88,964 → 1,030,106 → 2,136,974, tỉ lệ singleton tăng 45.7% → 73.1% → 87.9%, tỉ lệ n-gram chưa gặp trên validation tăng 0% → 24.2% → 64.2%.

## Câu 3 - Tại sao smoothing cần thiết?

Vì MLE gán xác suất 0 cho mọi n-gram chưa gặp, chỉ một token như vậy làm cả câu có xác suất 0 và perplexity bằng ∞ (bigram/trigram MLE trên validation/test). Smoothing lấy bớt khối xác suất của các n-gram đã gặp để chia cho n-gram chưa gặp, giúp model đánh giá được dữ liệu mới. Tuy nhiên Laplace với V lớn chia quá nhiều nên perplexity rất cao, cần các phương pháp tốt hơn như add-k, backoff, interpolation, Kneser-Ney.

## Câu 4 - Perplexity đo điều gì?

Perplexity là nghịch đảo của trung bình nhân xác suất model gán cho từng token, PP = exp(-1/N Σ log P(w_i|context_i)). Nó đo mức độ "bối rối" của model, tương đương số lựa chọn đều nhau mà model phải phân vân ở mỗi bước. Perplexity càng thấp thì model càng gán xác suất cao cho dữ liệu đánh giá, và chỉ so sánh được khi cùng dữ liệu, cùng preprocessing, cùng vocabulary.

## Câu 5 - Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.

Không. Perplexity chỉ đo khả năng dự đoán token trên một tập dữ liệu cụ thể, không đo tính mạch lạc, đúng sự thật hay hữu ích. Perplexity thấp có thể do overfitting (trigram MLE có train PPL 9.9 nhưng valid PPL ∞), do vocabulary nhỏ hoặc nhiều UNK, hoặc do model ưu tiên từ phổ biến và câu ngắn an toàn. Văn bản sinh ra vẫn có thể lặp lại, nhạt hoặc mất mạch lạc ở phạm vi dài.

## Câu 6 - N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?

- Chỉ nhìn n - 1 từ trước, không nắm được phụ thuộc xa (chủ ngữ ở đầu câu, chủ đề của đoạn văn).
- Mỗi từ là một ký hiệu rời rạc, không biết cat và dog tương tự nhau nên không khái quát được từ n-gram đã gặp sang n-gram chưa gặp.
- Không có ngữ nghĩa và kiến thức thế giới, chỉ dựa vào đếm tần suất bề mặt, nên gán xác suất 0 hoặc rất thấp cho câu hợp lý chưa gặp (after you leave), đôi khi xếp câu vô nghĩa cao hơn câu hợp lý (Trigram Laplace xếp banana computer quickly trên is useful for NLP).
- Số tham số tăng theo V^n, cần dữ liệu rất lớn và vẫn bị sparsity.

## Câu 7 - Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?

Không. Theo giả định Markov, trigram chỉ dùng 2 từ ngay trước: P(w_100|w_1, ..., w_99) ≈ P(w_100|w_98, w_99), 97 từ đầu bị bỏ qua hoàn toàn. Tăng n để bao được context dài là không khả thi vì sparsity, đây là động lực cho neural language model (RNN, attention, Transformer) biểu diễn toàn bộ context bằng vector dày đặc.

## Mục 23 - Context dài hơn có luôn tốt hơn không?

Không. Context dài hơn chỉ tốt hơn trên training set: train PPL MLE 1,434 → 125 → 9.9. Trên dữ liệu mới, số n-gram unique tăng (48,337 → 970,945 → 2,107,942 sau khi map UNK) nhưng tỉ lệ n-gram chưa gặp trên validation cũng tăng 0% → 24.2% → 64.2% (test: 0% → 23.7% → 63.7%). Kết quả là bigram và trigram MLE có valid/test PPL = ∞, còn với Laplace thì valid PPL tăng theo n: 1,238 → 3,717 → 19,810 (test: 1,193 → 3,590 → 19,612). Khoảng cách giữa train và valid/test lớn nhất ở trigram, là dấu hiệu overfitting. Context dài chỉ có lợi khi có đủ dữ liệu để ước lượng các n-gram dài hoặc khi dùng smoothing/backoff tốt hơn.

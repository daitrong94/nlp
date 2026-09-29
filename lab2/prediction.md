# Prediction trước experiment

Phần Prediction, Reason, Confidence ghi trước khi chạy experiments.ipynb. Phần Sau experiment bổ sung sau khi có kết quả.

## Prediction 1 - Khi chuyển unigram → bigram → trigram, vocabulary có tăng không?

Prediction: Không. Vocabulary giữ nguyên, chỉ số lượng n-gram khác nhau tăng.
Reason: Vocabulary là tập các từ (unigram type) của corpus, không phụ thuộc n. Tăng n chỉ tạo ra nhiều tổ hợp từ hơn từ cùng một tập từ.
Confidence: Cao (95%).

Sau experiment: Đúng. Cả ba model dùng chung V = 48,337 (từ có count ≥ 2 trong train, gồm EOS và UNK). Cái tăng là số n-gram unique: 88,964 → 1,030,106 → 2,136,974.

## Prediction 2 - Số lượng n-gram sẽ thay đổi như thế nào?

Prediction: Số n-gram unique tăng mạnh theo n, tỉ lệ n-gram chỉ xuất hiện một lần cũng tăng. Tổng số n-gram token gần như không đổi.
Reason: Số tổ hợp có thể có tăng theo V^n trong khi lượng dữ liệu cố định, mỗi câu độ dài L luôn sinh L + 1 n-gram (có padding) bất kể n, nên các n-gram bậc cao bị chia mỏng ra.
Confidence: Cao (85%).

Sau experiment: Đúng. Tổng n-gram token của cả ba model đều là 3,059,335. Unique: 88,964 / 1,030,106 / 2,136,974 (bigram gấp khoảng 11.6 lần unigram, trigram gấp khoảng 2.1 lần bigram). Tỉ lệ singleton: 45.7% / 73.1% / 87.9%.

## Prediction 3 - Mô hình nào có khả năng gặp zero probability nhiều hơn?

Prediction: Trigram (MLE).
Reason: Context gồm 2 từ nên có nhiều tổ hợp chưa từng xuất hiện trong train hơn bigram. Unigram gần như không gặp zero nếu từ lạ được map về UNK.
Confidence: Cao (90%).

Sau experiment: Đúng. Tỉ lệ token có P = 0 trên validation: unigram 0%, bigram 24.2%, trigram 64.2% (test: 0% / 23.7% / 63.7%). Perplexity MLE của bigram và trigram trên validation/test đều là ∞.

## Prediction 4 - Mô hình nào dự kiến có perplexity thấp hơn trên training set?

Prediction: Trigram có training perplexity thấp nhất, sau đó bigram, cao nhất là unigram.
Reason: Context dài hơn giúp model khớp dữ liệu train sát hơn. Nhiều trigram trong train là duy nhất nên P(w|h) gần 1.
Confidence: Trung bình - cao (75%), chưa chắc cho trường hợp có smoothing.

Sau experiment: Đúng với MLE: 1,434.2 → 125.3 → 9.9. Sai với Laplace: 1,436.2 → 3,111.6 → 11,887.8, tăng theo n. Lý do là V = 48,337 rất lớn so với C(h) của đa số context, add-one chuyển gần hết khối xác suất sang các n-gram chưa gặp, context càng dài thì C(h) càng nhỏ nên càng bị ảnh hưởng.

## Prediction 5 - Nếu corpus rất nhỏ, trigram có chắc chắn tốt hơn bigram không?

Prediction: Không. Với corpus nhỏ trigram thường tệ hơn bigram trên validation/test.
Reason: Dữ liệu ít thì phần lớn trigram trong validation/test chưa gặp, ước lượng cho context 2 từ dựa trên rất ít mẫu nên kém tin cậy (overfitting train).
Confidence: Cao (85%).

Sau experiment: Đúng. Với toàn bộ train, trigram Laplace có valid PPL 19,810 so với 3,717 của bigram; trigram MLE là ∞. Khi chỉ dùng 10% train, khoảng cách còn lớn hơn: 32,088 so với 11,605, tỉ lệ trigram chưa gặp trên validation lên tới 80.9% (bigram 45.7%).

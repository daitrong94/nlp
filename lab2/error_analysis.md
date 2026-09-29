# Error analysis

Các trường hợp lấy từ mục 10 của experiments.ipynb (300 vị trí ngẫu nhiên trong test set, dự đoán top-1 bằng Bigram/Trigram MLE). Top-1 accuracy: Bigram 10.3%, Trigram 10.0%. Top-5 accuracy: Bigram 25.0%, Trigram 18.3%.

## Prediction đúng 1

- Model: Trigram MLE
- Context: bigger drivers then you should anticipate the fact
- Model prediction: that
- Expected: that
- Probability: P(that|the fact) = 219/263 = 0.833

Giải thích: the fact that là cụm cố định rất phổ biến, context the fact xuất hiện 263 lần trong train và 83% trong số đó theo sau là that. Hai từ context đủ để xác định từ tiếp theo, bigram chỉ nhìn fact nên chỉ đạt P(that|fact) = 0.370.

## Prediction đúng 2

- Model: Bigram MLE
- Context: it was probably closer
- Model prediction: to
- Expected: to
- Probability: P(to|closer) = 62/114 = 0.544

Giải thích: closer to là collocation mạnh, một từ context đã đủ. Trigram lại không dự đoán được vì context probably closer chưa từng xuất hiện trong train (C(h) = 0), cho thấy context ngắn hơn đôi khi lại có lợi vì được ước lượng từ nhiều dữ liệu hơn.

## Prediction sai 1

- Model: Bigram MLE và Trigram MLE
- Context: after you
- Model prediction: Bigram: can (P = 0.112), Trigram: have (P = 0.145)
- Expected: leave
- Probability của expected: Bigram P(leave|you) = 27/26,947 = 0.001, Trigram P(leave|after you) = 0/62 = 0

Nguyên nhân: context quá ngắn và unseen n-gram. Bigram chỉ nhìn you nên dự đoán từ phổ biến nhất sau you (can) mà bỏ qua after. Trigram có đúng context nhưng after you chỉ xuất hiện 62 lần trong train và chưa lần nào đi với leave, nên MLE gán xác suất 0 (insufficient training data, sparsity) dù after you leave là cụm hoàn toàn tự nhiên.

## Prediction sai 2

- Model: Bigram MLE và Trigram MLE
- Context: how do i know it's time to transition
- Model prediction: Bigram: from (P = 0.189), Trigram: from (P = 0.333)
- Expected: marketing
- Probability của expected: Bigram P(marketing|transition) = 0/90 = 0, Trigram P(marketing|to transition) = 0/6 = 0

Nguyên nhân: unseen n-gram và sparsity. marketing có trong vocabulary nhưng chưa từng đứng sau transition trong train. Context to transition chỉ có 6 lần xuất hiện nên phân phối trigram gần như chỉ là ghi nhớ vài câu trong train. Ngoài ra transition ở đây là ngoại động từ có tân ngữ (transition marketing ...), cách dùng hiếm trong train, nơi transition chủ yếu đi với from/to. Không có smoothing nên xác suất đúng bằng 0.

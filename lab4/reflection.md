# Reflection

## Dataset có imbalance không? (mục 11)

Không. Train và test đều có 12,500 positive và 12,500 negative (50/50). Sau khi bỏ duplicate, tỉ lệ positive là 50.08% (train), 50.09% (validation), 50.30% (test). Vì vậy trên IMDb accuracy và F1 gần như bằng nhau, nhưng vẫn phải báo cáo đủ precision, recall, F1.

## Vì sao fit TF-IDF trên toàn bộ dataset là data leakage? (mục 12)

Vocabulary và IDF là tham số mà model học từ dữ liệu. Nếu fit trên cả test thì:

- vocabulary chứa các từ chỉ có trong test (ở IMDb: 17,940 term, vocabulary tăng từ 43,172 lên 61,112);
- IDF của mọi từ được tính có cả document của test.

Khi đó test không còn là dữ liệu "chưa thấy", nên kết quả trên test không ước lượng đúng hiệu năng khi deploy. Ở IMDb chênh lệch nhỏ (F1 0.8904 và 0.8901) vì train đủ lớn, nhưng với dataset nhỏ hoặc khi bước fit phức tạp hơn (chọn feature, scaling, Word2Vec) thì mức sai lệch có thể lớn. Trong notebook, vectorizer và Word2Vec đều nằm trong Pipeline, chỉ fit trên train.

Ngoài ra IMDb có 123 review trùng nhau giữa train và test, đây cũng là một dạng leakage nên đã bị loại khỏi test.

## Tại sao hai representation tạo ra kết quả khác nhau? (mục 16)

| Representation | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| TF-IDF | Logistic Regression | 0.890 | 0.891 | 0.889 | 0.890 |
| Avg Word2Vec | Logistic Regression | 0.860 | 0.868 | 0.850 | 0.859 |
| TF-IDF | Linear SVM | 0.890 | 0.891 | 0.890 | 0.891 |

- TF-IDF là sparse và lexical: mỗi term là một chiều riêng, nên model gán được weight lớn cho đúng các từ đặc trưng (worst -12.29, great +9.86, waste, 7/10). Với sentiment, các từ này gần như quyết định nhãn.
- Avg Word2Vec là dense và semantic nhưng học từ co-occurrence, nên good và bad có context giống nhau (cos = 0.709, bad nằm trong top 4 láng giềng của good). Thông tin "tích cực hay tiêu cực" không phải là thứ embedding học để phân biệt.
- Averaging làm loãng tín hiệu: review train dài trung bình 234 token, vài từ cảm xúc bị trộn với hàng trăm từ trung tính. Thứ tự từ cũng mất, nên "good is movie this" và "this movie is good" có cùng vector (cos = 1.000).
- Avg Word2Vec chỉ có 100 chiều so với 43,172 chiều của TF-IDF, ít thông tin hơn để classifier tách hai class.
- TF-IDF + LR và TF-IDF + Linear SVM gần như bằng nhau (F1 0.890 và 0.891), nên chênh lệch chủ yếu do representation chứ không do classifier.

## Experiment 4 - Representation có hiểu vai trò của not không?

| S1 | S2 | cos Avg W2V | P(pos) A: S1 / S2 | P(pos) B: S1 / S2 | P(pos) uni+bigram: S1 / S2 |
|---|---|---|---|---|---|
| this movie is good | this movie is not good | 0.976 | 0.825 / 0.370 | 0.994 / 0.714 | 0.932 / 0.055 |

Không. Với average embedding, S2 chỉ khác S1 ở chỗ cộng thêm vector của not rồi chia 5 thay vì 4, nên hai vector gần trùng nhau (cos = 0.976) và Model B vẫn đoán S2 là positive (0.714). Hoán vị từ cũng không đổi vector.

TF-IDF unigram đoán đúng S2 (0.370) nhưng không phải vì hiểu phủ định: not có weight âm (-1.87) chỉ vì nó hay xuất hiện trong review chê, còn good vẫn đóng góp +2.06. Bằng chứng là "not bad at all" bị cả ba model đoán negative (P(pos) ≤ 0.006), vì not và bad đều âm. Chỉ khi thêm bigram, cụm "not good" mới thành một feature riêng (-2.37).

## Experiment 5 - Vì sao bigram giúp classification nhưng làm feature space lớn hơn?

| Features | Vocabulary | Test F1 | Training time |
|---|---|---|---|
| Unigram | 43,172 | 0.890 | 3.9 s |
| Uni + Bigram | 365,203 | 0.907 | 16.1 s |

- Giúp: bigram giữ được một phần word order và phủ định cục bộ. "very good" (+4.57), "not good" (-2.37), "not bad" (+1.59), "the worst" (-11.73) là các feature riêng mà unigram không biểu diễn được. F1 tăng 1.7 điểm, S2 được đoán đúng với P(pos) = 0.055.
- Feature space lớn hơn: số bigram có thể có tăng theo bình phương vocabulary, thực tế vocabulary tăng 8.5 lần dù đã lọc min_df = 2. Phần lớn bigram hiếm nên weight ít tin cậy, dễ overfit, tốn bộ nhớ và train lâu hơn 4 lần. Bigram cũng chỉ thấy cặp liền kề, nên "not ... good" cách xa nhau vẫn không xử lý được.

## Giới hạn của bag-of-words và static embeddings

- TF-IDF không có semantic understanding: excellent và superb là hai chiều độc lập, model phải thấy từng từ trong train mới học được weight.
- Word2Vec không có contextual understanding: mỗi từ chỉ có một vector, "like" trong "I like this movie" và "a movie like this" giống nhau. Model A thậm chí đoán "i like this movie" là negative (P(pos) = 0.30).
- Cả hai đều không mô hình hoá thứ tự từ và phạm vi của phủ định, mỉa mai. Đây là lý do LAB 05 chuyển sang sequence model (RNN/LSTM/GRU).

## Individual learning check

- Tại sao accuracy 95% chưa chắc tốt: nếu 95% dữ liệu là negative, model luôn đoán negative cũng đạt 95% nhưng recall positive = 0, F1 = 0. Trên tập test 95/5 tạo từ IMDb, model này có accuracy 0.950, cao hơn TF-IDF + LR (0.891), dù TF-IDF + LR có recall 0.901 và F1 0.453.
- TF-IDF khác average Word2Vec ở điểm nào: TF-IDF là vector sparse biểu diễn document theo từng term, giữ được từ đặc trưng nhưng không biết từ đồng nghĩa. Average Word2Vec là vector dense, từ gần nghĩa có vector gần nhau, nhưng việc lấy trung bình làm mất từ quan trọng, mất thứ tự, và từ trái nghĩa cũng gần nhau.
- Tại sao "not good" có thể gây lỗi: bag-of-words tách thành not và good. good có weight dương lớn, not chỉ là một từ âm chung chung, model không biết not đang phủ định good. Với average embedding, not chỉ là thêm một vector vào trung bình.
- Data leakage xảy ra ở đâu nếu fit TF-IDF trước khi split: ở bước fit. Vocabulary và IDF đã dùng thông tin của test, nên khi đánh giá thì model đã "thấy" test một phần.
- Precision và recall khác nhau thế nào: precision = TP/(TP + FP), trong các mẫu model đoán positive có bao nhiêu đúng. Recall = TP/(TP + FN), trong các mẫu thật sự positive model tìm được bao nhiêu. Tăng threshold thường làm precision tăng, recall giảm.
- Tại sao thêm bigram có thể cải thiện sentiment classification: các cụm như "not good", "very good", "not bad" mang sentiment khác với từng từ riêng lẻ. Trên IMDb, F1 tăng từ 0.890 lên 0.907.

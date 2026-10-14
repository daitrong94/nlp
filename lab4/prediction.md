# Prediction

Prediction, Reason và Confidence được ghi trước khi chạy notebook. Dòng "Sau experiment" ghi sau khi có kết quả. Dataset: IMDb (sentiment, positive/negative).

## Mục 5 - TF-IDF hay Average Word2Vec phù hợp hơn với IMDb?

- Prediction: Hai representation với cùng Logistic Regression sẽ cho kết quả khác nhau. TF-IDF phù hợp hơn với IMDb.
- Reason: Sentiment trong review được quyết định bởi các từ đặc trưng (great, worst, waste). TF-IDF giữ riêng từng từ, còn trung bình embedding của cả review dài (khoảng 230 token) sẽ làm loãng các từ này. Ngoài ra good và bad có context giống nhau nên embedding gần nhau.
- Confidence: 75%
- Sau experiment: Đúng. Test F1: TF-IDF + LR 0.890, Avg Word2Vec + LR 0.859. Trong Word2Vec train trên IMDb, cos(good, bad) = 0.709 và bad nằm trong top 4 láng giềng của good.

## Prediction 1 - TF-IDF + Logistic Regression có hoạt động tốt không?

- Prediction: Có, accuracy và F1 khoảng 0.85 - 0.90.
- Reason: Dữ liệu cân bằng, đủ lớn (khoảng 20,000 review train), và review có nhiều từ mang cảm xúc rõ ràng.
- Confidence: 80%
- Sau experiment: Đúng. Test accuracy 0.890, precision 0.891, recall 0.889, F1 0.890, train mất 3.9 s.

## Prediction 2 - Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

- Prediction: Không. Average Word2Vec sẽ thấp hơn TF-IDF vài điểm F1.
- Reason: Averaging làm mất từ quan trọng và mất thứ tự từ. Embedding đặt từ trái nghĩa gần nhau. Chỉ có 100 chiều so với hơn 40,000 feature của TF-IDF.
- Confidence: 70%
- Sau experiment: Đúng. F1 0.859 so với 0.890, và chậm hơn nhiều (88.6 s gồm cả train Word2Vec, so với 3.9 s).

## Prediction 3 - Tăng vocabulary size thì accuracy có chắc chắn tăng không?

- Prediction: Không chắc. Accuracy tăng nhanh khi vocabulary còn nhỏ, sau đó gần như đứng yên, có thể giảm nhẹ vì thêm nhiều từ hiếm gây nhiễu.
- Reason: Các từ quan trọng nhất đều là từ phổ biến. Từ hiếm có ít mẫu nên weight không đáng tin.
- Confidence: 60%
- Sau experiment: Đúng một phần. Test accuracy theo max_features 100 / 500 / 1,000 / 5,000 / 20,000 / 43,172 là 0.729 / 0.843 / 0.866 / 0.884 / 0.887 / 0.890. Accuracy vẫn tăng nhưng bão hoà: từ 20,000 lên 43,172 từ chỉ thêm 0.3 điểm. Trong thí nghiệm này chưa thấy giảm, do min_df = 2 và regularization đã hạn chế ảnh hưởng của từ hiếm.

## Prediction 4 - Bỏ stopwords thì performance có chắc chắn tăng không?

- Prediction: Không, có thể giảm.
- Reason: Danh sách stopword của sklearn có not, no, never, nothing, very, too, but, well. Đây là các từ mang tín hiệu cảm xúc hoặc phủ định.
- Confidence: 70%
- Sau experiment: Đúng. Test F1 giảm từ 0.890 xuống 0.883. Giữ lại các từ phủ định thì được 0.885, vẫn thấp hơn không bỏ, vì các từ như well, very, but, too cũng có ích.

## Prediction 5 - Accuracy và F1 có thể cho kết luận khác nhau không?

- Prediction: Có, khi dữ liệu mất cân bằng. Với IMDb cân bằng thì hai metric gần như bằng nhau.
- Reason: Accuracy bị class đa số chi phối, F1 chỉ tính trên positive class.
- Confidence: 85%
- Sau experiment: Đúng. Trên IMDb, accuracy và F1 chênh nhau dưới 0.001 ở mọi model. Trên tập test được tạo lại theo tỉ lệ 95/5 (12,266 negative, 646 positive): model luôn đoán negative có accuracy 0.950, F1 = 0; TF-IDF + LR có accuracy thấp hơn (0.891) nhưng F1 = 0.453 và recall = 0.901. Nếu chỉ nhìn accuracy thì sẽ chọn sai model.

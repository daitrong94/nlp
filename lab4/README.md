# LAB 04 - Text Classification

## Files

- calculations.md: Bài 1 - 4 (logit, sigmoid, threshold, confusion matrix, accuracy trap)
- prediction.md: prediction trước experiment và đối chiếu sau experiment
- preprocessing.py: load IMDb, clean_text, tokenize, dataset inspection, deduplicate, split train/validation, MeanEmbeddingVectorizer (Average Word2Vec), unit test
- classification.ipynb: Experiment 1 - 5, data leakage, evaluation, error analysis, inspect weights
- results.csv: số liệu do notebook ghi ra (experiment, model, setting, split, metric, value)
- confusion_matrix.png: confusion matrix trên test của 4 model
- error_analysis.md: confusion matrix, 3 review đúng, 6 review sai, top features
- reflection.md: data leakage, so sánh representation, word order, n-gram, learning check

## Cách chạy

```
pip install numpy pandas scikit-learn gensim matplotlib pyarrow datasets jupyter
python preprocessing.py        # unit test
```

Sau đó chạy toàn bộ classification.ipynb (khoảng 7 phút). Dữ liệu đọc từ ../imdb/imdb_train.parquet và ../imdb/imdb_test.parquet (thư mục imdb ngang hàng với lab4). Nếu chưa có, load_imdb() tự tải từ Hugging Face (stanfordnlp/imdb, bản sao của Large Movie Review Dataset, ai.stanford.edu/~amaas/data/sentiment).

## Dataset và split

| | Train | Test |
|---|---|---|
| Samples | 25,000 | 25,000 |
| Negative / Positive | 12,500 / 12,500 | 12,500 / 12,500 |
| Missing text / label | 0 / 0 | 0 / 0 |
| Duplicate | 96 | 199 |
| Độ dài (token): mean / median / max | 233.6 / 174 / 2,473 | 228.3 / 172 / 2,246 |

- Dataset cân bằng, không có missing. Có 123 review trùng giữa train và test.
- Bỏ duplicate, bỏ review test đã có trong train, rồi chia train thành train/validation 80/20 (stratified, seed 42): train 19,923, validation 4,981, test 24,678.
- Mọi bước fit (TF-IDF, Word2Vec, StandardScaler) chỉ fit trên train. C được chọn theo F1 trên validation, test chỉ dùng để báo cáo.

## Thiết lập

- Preprocessing: bỏ thẻ HTML, lowercase, tokenize bằng regex [a-z0-9]+('[a-z]+)?, giữ stopword.
- TF-IDF: min_df = 2, sublinear tf.
- Average Word2Vec: Skip-gram train trên train split, vector_size = 100, window = 5, min_count = 5, epochs = 10. Vector document là trung bình word vector, chuẩn hoá bằng StandardScaler.
- Classifier: LogisticRegression và LinearSVC của sklearn. Positive class = positive.

## Kết quả chính (test)

| Representation | Classifier | Vocabulary | Train time | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| TF-IDF unigram | Logistic Regression (baseline) | 43,172 | 3.9 s | 0.890 | 0.891 | 0.889 | 0.890 |
| Avg Word2Vec | Logistic Regression | 27,006 từ, 100 chiều | 88.6 s | 0.860 | 0.868 | 0.850 | 0.859 |
| TF-IDF unigram | Linear SVM | 43,172 | 3.7 s | 0.890 | 0.891 | 0.890 | 0.891 |
| TF-IDF uni + bigram | Logistic Regression | 365,203 | 16.1 s | 0.907 | 0.905 | 0.910 | 0.907 |

Train time của Avg Word2Vec gồm 79.6 s train Word2Vec.

- TF-IDF tốt hơn Avg Word2Vec 3.1 điểm F1. LR và Linear SVM gần như bằng nhau.
- Bigram tăng F1 thêm 1.7 điểm nhưng vocabulary lớn gấp 8.5 lần.
- Bỏ stopwords làm F1 giảm (0.890 → 0.883). Tăng vocabulary làm F1 tăng nhưng bão hoà từ khoảng 20,000 từ.
- "this movie is not good": Avg Word2Vec + LR vẫn đoán positive (0.714), TF-IDF uni + bigram đoán đúng (0.055).
- Lỗi điển hình: negation, sarcasm, review khen/chê lẫn lộn và nhãn nhiễu.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated content: preprocessing.py, classification.ipynb, results.csv, confusion_matrix.png, bản nháp calculations.md, prediction.md, error_analysis.md, reflection.md
- Verification: unit test trong preprocessing.py, kiểm tra z = wᵀx + b và σ(z) tự tính khớp với decision_function và predict_proba, đối chiếu kết quả tính tay với code, đối chiếu số liệu trong các file .md với output của notebook

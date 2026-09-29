# LAB 02 - Language Models

Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng, HK I - 2026
Chủ đề: N-gram Language Models, Smoothing and Perplexity

## Cấu trúc thư mục

- calculations.md: bài tính tay (mục 7, 9, 11, 18)
- prediction.md: prediction trước experiment và đối chiếu sau experiment (mục 12)
- ngram_lm.py: core implementation, class NGramLanguageModel (unigram/bigram/trigram, MLE và Laplace, log probability, perplexity, next-word prediction, sentence ranking) và các hàm build_vocabulary, count_ngrams, train_unigram, train_bigram, train_trigram, probability, sentence_probability, sentence_log_probability
- experiments.ipynb: Experiment 1 - 3, log probability, context length, corpus size, next-word prediction, sentence ranking (mục 13, 15, 16, 19 - 23)
- results.csv: perplexity và prediction results (sinh ra từ notebook)
- error_analysis.md: 2 prediction đúng, 2 prediction sai (mục 22)
- reflection.md: câu hỏi tổng kết (mục 24) và câu hỏi context length (mục 23)

## Cách chạy

1. Cài dependencies: pip install numpy pandas matplotlib jupyter
2. Chạy unit test: python ngram_lm.py
3. Mở experiments.ipynb và chạy toàn bộ. Notebook ghi results.csv.

Đường dẫn corpus và các tham số cấu hình ở cell Setup (CORPUS_PATH, N_DOCS, MIN_COUNT, SEED).

## Thiết kế

- Corpus: C4 (c4-train.00000-of-01024-30K.json), dùng 10,000 document đầu, xáo trộn với seed 42, chia theo document train/valid/test = 80/10/10.
- Preprocessing: tách câu theo . ! ? và xuống dòng, tokenize lowercase bằng regex [a-z0-9]+('[a-z]+)?.
- Token đặc biệt: BOS (bắt đầu câu), EOS (kết câu), UNK (từ ngoài vocabulary),, xem giá trị cụ thể ở các hằng số cùng tên trong ngram_lm.py.
- Boundary tokens: mỗi câu được thêm n - 1 token BOS ở đầu và EOS ở cuối. BOS chỉ làm context, không được dự đoán.
- Vocabulary: build từ train, từ có tần suất < MIN_COUNT (= 2) được thay bằng UNK. Mọi model dùng chung vocabulary để perplexity so sánh được. V = số từ + EOS + UNK = 48,337.
- MLE: P(w|h) = C(h, w)/C(h), với C(h) = Σ_w C(h, w). Context chưa gặp thì P = 0.
- Laplace: P(w|h) = (C(h, w) + 1)/(C(h) + V).
- Perplexity: exp(-1/N Σ log P(w_i|context_i)), N là số token được dự đoán (gồm EOS). Nếu có token P = 0 thì perplexity = ∞ (notebook báo thêm zero_rate).

Lưu ý: do có BOS, xác suất câu trong code là P(the|BOS) P(cat|the) ... chứ không phải P(the) P(cat|the) ... như Bài 3 trong đề, nên số liệu tính tay và code có thể lệch ở từ đầu tiên.

## Kết quả chính

| Model | Smoothing | Train PPL | Valid PPL | Test PPL |
|---|---|---|---|---|
| Unigram | MLE | 1,434.2 | 1,233.4 | 1,188.2 |
| Unigram | Laplace | 1,436.2 | 1,238.1 | 1,193.1 |
| Bigram | MLE | 125.3 | ∞ | ∞ |
| Bigram | Laplace | 3,111.6 | 3,717.2 | 3,589.7 |
| Trigram | MLE | 9.9 | ∞ | ∞ |
| Trigram | Laplace | 11,887.8 | 19,809.9 | 19,612.3 |

Chi tiết và giải thích xem experiments.ipynb, prediction.md và reflection.md.

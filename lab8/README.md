# LAB 08 - Pretrained Transformers và BERT Fine-Tuning

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- pretrained_classification.py: `BertClassifier` với 3 chế độ (fine-tune, frozen, from scratch), hàm `encode`, `train`, `predict_logits`, có unit test
- tokenization.ipynb: special tokens, attention mask, subword, thống kê độ dài trên IMDb, MLM với BERT-mini
- fine_tuning.ipynb: Experiment 1 - 5, confusion matrix, confidence, application, error analysis
- results.csv, data_efficiency.png, confusion_matrix.png, confidence_analysis.png

## Cách chạy

```
pip install numpy pandas scikit-learn torch transformers matplotlib pyarrow datasets jupyter
python pretrained_classification.py
```

Sau đó chạy tokenization.ipynb (vài phút) và fine_tuning.ipynb (khoảng 3 giờ 15 phút trên CPU).

- Model, tokenizer và dữ liệu ngoài domain nằm ở `../hf_models` (`HF_HOME`) và được load offline (`HF_HUB_OFFLINE=1`). Thư mục này không được đưa lên git. Có thể tải lại bằng `huggingface_hub`.
- Dữ liệu ngoài domain: `fancyzhx/amazon_polarity`, `fancyzhx/yelp_polarity` (mỗi tập 2,000 mẫu cân bằng) và `stanfordnlp/sst2` (872 câu validation).
- IMDb dùng `../lab4/preprocessing.py` và `../imdb/`. Đặt `LAB8_QUICK=1` để chạy thử trên tập con.

## Reproducibility

```
Model:               google/bert_uncased_L-4_H-256_A-4 (BERT-mini: 4 layer, hidden 256, 4 head, 11.17M tham số)
Tokenizer:           bert-base-uncased (WordPiece, 30,522 token)
Dataset:             IMDb (stanfordnlp/imdb), bỏ trùng lặp, cùng split với LAB 04 - 07
Train split:         19,923 review (subset 10/25/50% lồng nhau, stratified)
Validation split:    4,981 review
Test split:          24,678 review (Exp 3 - 4 dùng test-5k cố định)
Max sequence length: 128 (Exp 4: 64 / 128 / 256)
Batch size:          32
Learning rate:       1e-4 (frozen head: Adam 1e-3, 30 epoch)
Epochs:              2, giữ checkpoint có validation loss thấp nhất
Optimizer:           AdamW (weight decay 0.01), linear warmup 10% + decay, clip 1.0
Random seed:         42
Hardware:            CPU Intel 8 thread, không GPU, torch 2.13.0+cpu
Training time:       fine-tune toàn bộ 36 phút, from scratch 35 phút, frozen 8 phút; cả notebook 3 giờ 15 phút
```

## Kết quả chính

| Model (test 24,678 review) | Accuracy | Precision | Recall | F1 | Train time | Trainable params |
|---|---|---|---|---|---|---|
| From scratch | 0.815 | 0.824 | 0.805 | 0.814 | 2,070 s | 11,171,074 |
| Frozen encoder ([CLS]) | 0.740 | 0.752 | 0.721 | 0.736 | 502 s | 514 |
| Pretrained + fine-tune | 0.842 | 0.845 | 0.841 | **0.843** | 2,171 s | 11,171,074 |

| F1 theo training data (test-5k) | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Pretrained | 0.788 | 0.806 | 0.827 | 0.841 |
| From scratch | 0.131 | 0.711 | 0.786 | 0.808 |

| Max length (25% data) | 64 | 128 | 256 |
|---|---|---|---|
| F1 | 0.756 | 0.806 | 0.853 |
| s/epoch | 168 | 339 | 763 |
| Attention memory (tính theo công thức) | 8 MB | 32 MB | 128 MB |

| F1 khi đổi domain | IMDb | SST-2 | Amazon | Yelp |
|---|---|---|---|---|
| Pretrained | 0.843 | 0.800 | 0.831 | 0.823 |
| From scratch | 0.814 | 0.727 | 0.790 | 0.765 |

- Pretrained tốt hơn from scratch 2.9 điểm F1 khi dùng đủ dữ liệu, và hơn 65.7 điểm khi chỉ có 10% dữ liệu.
- Fine-tuning tốt hơn frozen 10.7 điểm F1. Frozen rẻ hơn khoảng 4 lần.
- Tăng max_length giúp F1 trên IMDb vì review dài, nhưng chi phí attention tăng theo $T^2$.
- F1 giảm trên mọi domain khác, giảm nhiều nhất trên SST-2 (câu ngắn).
- Confidence cao không đảm bảo đúng: 25.6% lỗi có confidence ≥ 0.9.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: pretrained_classification.py, tokenization.ipynb, fine_tuning.ipynb, results.csv, các hình .png, bản nháp các file .md
- Modified: không
- Verified: unit test (keys của tokenizer, attention mask, số tham số, frozen chỉ còn 514 tham số train được, padding không đổi output, trọng số pretrained khác trọng số ngẫu nhiên), bài tính MLM và tokenization kiểm lại trong tokenization.ipynb, đối chiếu số liệu trong các file .md với output của notebook

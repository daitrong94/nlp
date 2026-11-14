# LAB 09 - Named Entity Recognition và Information Extraction

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- ner_data.py: load CoNLL-2003, tách entity từ chuỗi BIO, entity-level và token-level metric tự viết, có unit test
- baseline_ner.py: token features → Logistic Regression
- tokenizer_alignment.ipynb: word_ids, label alignment (Strategy A và B), Experiment 1
- ner_finetuning.ipynb: fine-tune DistilBERT, Experiment 2 - 6, confusion matrix, context analysis
- information_extraction.py: raw text → NER → JSON, đánh giá trên 30 câu tin tức có gold annotation
- results.csv, entity_metrics.csv, entity_length_analysis.png, confusion_matrix.png

## Cách chạy

```
pip install numpy pandas scikit-learn torch transformers matplotlib pyarrow jupyter
python ner_data.py
python baseline_ner.py
```

Sau đó chạy tokenizer_alignment.ipynb (1 phút), ner_finetuning.ipynb (khoảng 45 phút trên CPU), rồi:

```
python information_extraction.py
python information_extraction.py "Barack Obama visited Paris."
```

- Model, dataset và model đã fine-tune nằm ở `../hf_models` (`HF_HOME`), load offline (`HF_HUB_OFFLINE=1`). Thư mục này không được đưa lên git. Cần tải lại `distilbert/distilbert-base-cased` và dataset `lhoestq/conll2003` bằng `huggingface_hub.snapshot_download`.
- Đặt `LAB9_QUICK=1` để chạy thử notebook trên tập con.

## Reproducibility

```
Model:               distilbert/distilbert-base-cased (6 layer, hidden 768, 12 head, 65.2M tham số)
Tokenizer:           WordPiece cased, 28,996 token
Dataset:             CoNLL-2003 (lhoestq/conll2003), 9 nhãn BIO
Train / val / test:  14,041 / 3,250 / 3,453 câu
Label alignment:     Strategy A (token đầu của word mang nhãn, subword sau = -100)
Max sequence length: 128 (7 câu bị cắt trong cả ba tập)
Batch size:          32, gom theo độ dài
Learning rate:       5e-5
Epochs:              3, giữ checkpoint có entity F1 cao nhất trên validation (epoch 3)
Optimizer:           AdamW (weight decay 0.01), linear warmup 10% + decay, clip 1.0
Random seed:         42
Hardware:            CPU Intel 8 thread, không GPU, torch 2.13.0+cpu
Training time:       Transformer 40 phút, baseline 2 phút
```

## Kết quả chính (test, entity-level)

| Model | Precision | Recall | F1 | Token accuracy |
|---|---|---|---|---|
| Baseline (token features + LR) | 0.753 | 0.795 | 0.774 | 0.962 |
| DistilBERT fine-tune | 0.885 | 0.903 | **0.894** | 0.979 |

| F1 theo loại | PER | LOC | ORG | MISC |
|---|---|---|---|---|
| Baseline | 0.803 | 0.847 | 0.676 | 0.761 |
| Transformer | 0.951 | 0.922 | 0.870 | 0.763 |

| F1 theo độ dài entity | 1 word | 2 word | 3 word | 4+ word |
|---|---|---|---|---|
| Baseline | 0.756 | 0.836 | 0.649 | 0.468 |
| Transformer | 0.890 | 0.908 | 0.854 | 0.862 |

| Recall theo số lần gặp trong train | 0 | 1 - 2 | 3 - 9 | 10+ |
|---|---|---|---|---|
| Baseline | 0.664 | 0.779 | 0.893 | 0.962 |
| Transformer | 0.863 | 0.885 | 0.923 | 0.965 |

| Information Extraction (30 câu, 75 entity) | Precision | Recall | F1 |
|---|---|---|---|
| PERSON / ORGANIZATION / LOCATION | 0.959 | 0.933 | 0.946 |

- Token accuracy cao (0.96 - 0.98) vì 83% token là O. Phải đánh giá ở mức entity.
- Lợi thế của Transformer nằm ở entity chưa gặp, entity dài và chuỗi mang nhiều loại (đúng 85.8% so với 74.1%).
- Entity bị tách thành subword nhiều gấp 4 lần non-entity (45% so với 11%). Headline in hoa toàn bộ làm recall giảm từ 0.915 xuống 0.806.
- MISC khó nhất và không cải thiện. ORG ↔ LOC là cặp nhầm nhiều nhất.
- IE: 26 trên 30 câu đúng hoàn toàn. Câu viết thường toàn bộ sai cả 3 entity.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: ner_data.py, baseline_ner.py, information_extraction.py (gồm 30 câu và gold annotation), tokenizer_alignment.ipynb, ner_finetuning.ipynb, results.csv, entity_metrics.csv, các hình .png, bản nháp các file .md
- Modified: không
- Verified: unit test (entity metric trên ví dụ "New York" của đề, quy ước conlleval, bộ tách từ), alignment khôi phục đúng nhãn gốc 14,040/14,041 câu train, baseline cho cùng kết quả khi chạy riêng và trong notebook, đối chiếu số liệu trong các file .md với output của notebook

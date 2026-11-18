# LAB 09 - NER và Information Extraction

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- ner_data.py: load CoNLL-2003, entity-level metric (có unit test)
- baseline_ner.py: token features → Logistic Regression
- tokenizer_alignment.ipynb: word_ids, label alignment, Experiment 1
- ner_finetuning.ipynb: fine-tune DistilBERT, Experiment 2 - 6
- information_extraction.py: text → NER → JSON, đánh giá trên 30 câu tin tức
- results.csv, entity_metrics.csv, entity_length_analysis.png, confusion_matrix.png

## Cách chạy

```
pip install numpy pandas scikit-learn torch transformers matplotlib pyarrow jupyter
python ner_data.py
python baseline_ner.py
```

Chạy tokenizer_alignment.ipynb, rồi ner_finetuning.ipynb (khoảng 45 phút trên CPU), sau đó chạy `python information_extraction.py`.

Model và dataset được load offline từ `../hf_models` (thư mục này không đưa lên git). Đặt `LAB9_QUICK=1` để chạy thử nhanh.

## Reproducibility

```
Model:          distilbert/distilbert-base-cased (65.2M tham số)
Dataset:        CoNLL-2003, train / val / test = 14,041 / 3,250 / 3,453 câu
Alignment:      Strategy A (token đầu của word mang nhãn, subword sau = -100)
Max length:     128
Batch / LR:     32 / 5e-5, AdamW, warmup 10%
Epochs:         3 (chọn epoch có val F1 cao nhất)
Seed:           42
Hardware:       CPU Intel 8 thread, không GPU
Training time:  Transformer 40 phút, baseline 2 phút
```

## Kết quả (test, entity-level)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.753 | 0.795 | 0.774 |
| DistilBERT | 0.885 | 0.903 | **0.894** |

| F1 theo loại | PER | LOC | ORG | MISC |
|---|---|---|---|---|
| Baseline | 0.803 | 0.847 | 0.676 | 0.761 |
| DistilBERT | 0.951 | 0.922 | 0.870 | 0.763 |

- Token accuracy cao (0.96 - 0.98) vì 83% token là O, nên phải đánh giá ở mức entity.
- Transformer mạnh hơn rõ ở entity chưa gặp, entity dài và từ đổi loại theo ngữ cảnh.
- IE trên 30 câu tin tức: F1 0.946.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: toàn bộ code, notebook, kết quả và bản nháp các file .md
- Modified: không
- Verified: unit test cho metric và alignment, đối chiếu số liệu trong các file .md với output của notebook

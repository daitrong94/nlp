# LAB 10 - Extractive Question Answering

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- qa_model.py: load SQuAD, char span → token span, sliding window, postprocess, EM/F1, baseline, class `QAModel` (có unit test)
- tokenizer_alignment.ipynb: offset mapping, sliding window, các trường hợp đặc biệt
- qa_finetuning.ipynb: baseline, fine-tune DistilBERT, Experiment 1 - 5, controlled test, error analysis, document QA
- document_qa.py: hỏi đáp trên một FAQ của học phần (515 token), 20 câu hỏi có gold
- results.csv, learning_curves.png, em_f1_analysis.png, answer_length_analysis.png

## Cách chạy

```
pip install numpy pandas torch transformers matplotlib pyarrow jupyter
python qa_model.py                 # unit test
python document_qa.py "When is the LAB 10 deadline?"
```

- qa_finetuning.ipynb được chạy trên **Google Colab (T4 GPU)**: upload notebook cùng `qa_model.py` và `document_qa.py`, rồi Run all (khoảng 30 - 40 phút).
- Chạy trên máy thì model và dataset được load offline từ `../hf_models` (thư mục này không đưa lên git). Đặt `LAB10_QUICK=1` để chạy thử nhanh trên CPU.

## Reproducibility

```
Model:          distilbert/distilbert-base-cased
Tokenizer:      distilbert/distilbert-base-cased (WordPiece, cased)
Dataset:        SQuAD v1.1
Train / val:    85,548 / 2,000 câu hỏi (val tách từ train theo bài viết)
Test:           SQuAD validation, 10,570 câu hỏi
Max length:     384, stride 128
Batch / LR:     32 / 3e-5, AdamW (weight decay 0.01), warmup 10%
Epochs:         2
Seed:           42
Hardware:       Google Colab, GPU Tesla T4, fp16
Training time:  14.7 phút (GPU memory peak 3.6 GB)
```

## Kết quả (test)

| Model | EM | F1 |
|---|---|---|
| Baseline (trùng từ) | 8.9 | 22.4 |
| DistilBERT | 74.6 | **82.9** |

| Controlled test (1000 câu) | F1 |
|---|---|
| original | 86.6 |
| + irrelevant | 86.1 |
| + distractor | **23.7** |
| reordered | 84.7 |

- Answer càng dài càng khó: EM giảm từ 79.8 (1 từ) xuống 52.2 (7+ từ).
- Overlap thấp giữa câu hỏi và context: F1 75.8. Overlap cao: 88.9.
- Context length (20K câu, 1 epoch): 128 / 256 / 384 cho F1 69.8 / 63.7 / 63.1. Kết quả 128 cao hơn vì có gấp đôi số window, tức gấp đôi số step train.
- Document QA: EM 75.0, F1 84.1 trên 20 câu hỏi.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: toàn bộ code, notebook, kết quả và bản nháp các file .md
- Modified: không
- Verified: unit test cho alignment và metric, đối chiếu số liệu trong các file .md với output của notebook

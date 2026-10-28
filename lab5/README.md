# LAB 05 - Sequence Models: RNN, LSTM, GRU

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- rnn_cell.py: SimpleRNNCell bằng NumPy, có unit test so với Bài 1 và PyTorch
- sequence_classification.ipynb: toàn bộ experiments
- results.csv, learning_curves.png, confusion_matrix.png, cùng các hình phụ sequence_length*.png và gradient_by_position.png

## Cách chạy

```
pip install numpy pandas scikit-learn torch matplotlib pyarrow datasets jupyter
python rnn_cell.py        # unit test
```

Sau đó chạy sequence_classification.ipynb (khoảng 35 phút trên CPU). Notebook dùng `../lab4/preprocessing.py` và dữ liệu ở `../imdb/`. Đặt `LAB5_QUICK=1` để chạy thử trên tập con.

## Thiết lập

- IMDb, cùng split với LAB 04: train 19,923 / validation 4,981 / test 24,678.
- Vocabulary 20,000 từ + PAD + UNK, max_length 200 (giữ token đầu).
- Embedding 100 → RNN / LSTM / GRU (hidden 128) → Linear. Adam 1e-3, batch 64, 8 epoch, giữ checkpoint có validation loss thấp nhất.
- Baseline: TF-IDF + Logistic Regression (LAB 04).

## Kết quả chính (test)

| Model | Train time | Accuracy | F1 |
|---|---|---|---|
| TF-IDF + LR | 3.6 s | 0.890 | **0.890** |
| RNN | 243 s | 0.750 | 0.751 |
| LSTM | 294 s | 0.825 | 0.827 |
| GRU | 776 s | 0.849 | 0.854 |

| LSTM, max_length | 50 | 100 | 200 | 100 (giữ token cuối) |
|---|---|---|---|---|
| F1 | 0.742 | 0.773 | 0.827 | 0.838 |

- LSTM và GRU tốt hơn RNN rõ rệt, nhưng không vượt TF-IDF. Khoảng cách nằm ở review dài.
- Sequence model có dùng thứ tự từ, nhưng chưa ổn định.
- Vanishing gradient: gradient ở token đầu của RNN chỉ còn khoảng $10^{-10}$ lần token cuối.
- Giữ phần cuối review tốt hơn giữ phần đầu: kết luận thường nằm ở cuối.
- Cả ba model overfit sau epoch 4 - 5.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: rnn_cell.py, sequence_classification.ipynb, results.csv, các hình .png, bản nháp các file .md
- Modified: không
- Verified: unit test trong rnn_cell.py (khớp PyTorch, sai số 1e-16), đối chiếu số liệu trong các file .md với output của notebook

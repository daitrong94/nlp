# LAB 07 - Transformer Architecture

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- positional_encoding.py: sinusoidal PE, có unit test
- transformer_block.py: Multi-Head Self-Attention, FFN, Transformer block tự ghép, có unit test (khớp `nn.TransformerEncoderLayer` khi dùng chung weight)
- transformer_classification.ipynb: toàn bộ experiments
- results.csv, positional_encoding.png, attention_heads.png, learning_curves.png, confusion_matrix.png

## Cách chạy

```
pip install numpy scipy pandas scikit-learn torch matplotlib pyarrow datasets jupyter
python positional_encoding.py
python transformer_block.py
```

Sau đó chạy transformer_classification.ipynb (khoảng 1 giờ 45 phút trên CPU). Notebook dùng `../lab4/preprocessing.py` và dữ liệu ở `../imdb/`. Đặt `LAB7_QUICK=1` để chạy thử trên tập con.

## Thiết lập

- IMDb, cùng split với LAB 04 - 06. Vocabulary 20,000 từ + PAD, UNK, CLS. Max_length 128 (giữ token đầu).
- **Transformer:** Embedding 128 → sinusoidal PE → 1 Transformer block (4 head, $d_{ff}$ = 256, post-norm) → mean pooling → Linear.
- **LSTM:** Embedding 128 → LSTM (hidden 128) → hidden state cuối → Linear.
- Cùng batch 64, Adam 1e-3, 5 epoch, clip 1.0, giữ checkpoint có validation loss thấp nhất.

## Kết quả chính (test)

| Model | Params | s/epoch | Accuracy | F1 |
|---|---|---|---|---|
| LSTM | 2,692,609 | 41 | 0.811 | **0.818** |
| Transformer (PE, mean) | 2,692,993 | 122 | 0.815 | 0.811 |
| Transformer (không PE) | 2,692,993 | 117 | 0.816 | **0.818** |
| Transformer (CLS) | 2,692,993 | 144 | 0.798 | 0.792 |

| Số head | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| $d_k$ | 128 | 64 | 32 | 16 |
| F1 | 0.788 | 0.802 | 0.811 | 0.809 |

| max_length | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Attention memory (batch 64, 4 head) | 1 MB | 4 MB | 16 MB | 64 MB |
| F1 | 0.695 | 0.760 | 0.811 | 0.841 |

- Không có PE thì Transformer không phân biệt được thứ tự ("dog bites man" và "man bites dog" giống hệt nhau). Trên IMDb, PE không làm tăng F1 vì sentiment chủ yếu dựa vào từ khoá.
- Tăng head không tăng tham số. 4 head tốt hơn 1 head, 8 head không tốt hơn 4 head. Các head học pattern khác nhau, có head gần như thừa.
- Transformer và LSTM gần bằng nhau, nhưng trên CPU Transformer chậm hơn 3 lần.
- Tăng max_length giúp F1 nhưng attention memory tăng theo $T^2$.
- Mean pooling tốt hơn CLS pooling sau 5 epoch.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: positional_encoding.py, transformer_block.py, transformer_classification.ipynb, results.csv, các hình .png, bản nháp các file .md
- Modified: không
- Verified: unit test (PE khớp tính tay; block khớp `nn.MultiheadAttention` và `nn.TransformerEncoderLayer` khi dùng chung weight; shape đúng như đề), bài tính kiểm lại trong notebook, đối chiếu số liệu trong các file .md với output của notebook

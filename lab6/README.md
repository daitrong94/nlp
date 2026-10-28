# LAB 06 - Attention Mechanism

## Files

- calculations.md, prediction.md, error_analysis.md, reflection.md
- attention_numpy.py: softmax ổn định số, `scaled_dot_product_attention`, `self_attention`, unit test
- self_attention.ipynb: toàn bộ experiments
- results.csv, attention_heatmap.png, confusion_matrix.png, cùng các hình phụ

## Cách chạy

```
pip install numpy scipy pandas scikit-learn torch gensim matplotlib pyarrow datasets jupyter
python attention_numpy.py        # unit test
```

Sau đó chạy self_attention.ipynb (khoảng 20 - 25 phút trên CPU). Notebook dùng `../lab4/preprocessing.py` và dữ liệu ở `../imdb/`. Đặt `LAB6_QUICK=1` để chạy thử trên tập con.

## Thiết lập

- IMDb, cùng split với LAB 04 - 05. Max_length 200, vocabulary 20,002.
- Experiment 1 - 3 dùng Word2Vec (d = 100) train trên train split.
- Ba classifier, cùng cách train như LAB 05:
  - **LSTM:** dùng hidden state cuối.
  - **Self-Attention:** $H = \operatorname{softmax}(QK^\top / \sqrt{128})V$, rồi pooling $c = \sum_t \beta_t h_t$.
  - **Self-Attention + pos:** như trên, cộng thêm positional embedding.
- Self-attention viết tay, không dùng `nn.MultiheadAttention`.

## Kết quả chính

| Model | Train time | Accuracy | F1 |
|---|---|---|---|
| TF-IDF + LR | 5 s | 0.890 | **0.890** |
| LSTM | 389 s | 0.825 | 0.827 |
| Self-Attention | 270 s | 0.837 | 0.842 |
| Self-Attention + pos | 300 s | 0.833 | 0.834 |

- Attention phụ thuộc query: cùng một câu, query sentiment chọn "good" (0.38), query topic chọn "movie" (0.24).
- Temperature thấp làm attention tập trung, cao làm attention phẳng. Không chia $\sqrt{d_k}$ thì softmax bão hoà khi $d_k$ lớn.
- Self-attention không có vị trí không phân biệt được thứ tự từ. Thêm positional embedding thì được.
- Attention tốt hơn LSTM ở review dài và không thiên về cuối câu.
- Chi phí tăng theo $T^2$: attention nhanh hơn LSTM khi T ≤ 1000, chậm hơn từ khoảng T = 2000.
- Heatmap không chứng minh model hiểu ngữ pháp. Attention chỉ trùng với token quan trọng thật trong khoảng một nửa số trường hợp.

Lưu ý: Word2Vec train đa luồng nên số liệu Experiment 1 - 3 có thể lệch nhẹ giữa các lần chạy.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: attention_numpy.py, self_attention.ipynb, results.csv, các hình .png, bản nháp các file .md
- Modified: không
- Verified: unit test trong attention_numpy.py (khớp tính tay và PyTorch), self-attention trong PyTorch khớp bản NumPy (sai số 1e-16), đối chiếu số liệu trong các file .md với output của notebook

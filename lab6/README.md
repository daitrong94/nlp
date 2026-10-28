# LAB 06 - Attention Mechanism

## Files

- calculations.md: Bài 1 - 4 (scores, softmax, context, toàn bộ attention), scaling, temperature, numerical stability, bài self-attention ["I", "like", "cats"], kích thước attention matrix
- prediction.md: prediction trước experiment và đối chiếu sau experiment
- attention_numpy.py: softmax ổn định số, entropy, `scaled_dot_product_attention(Q, K, V)` trả về (scores, attention_weights, context), `self_attention`, unit test
- self_attention.ipynb: kiểm tra bài tính, Experiment 1 - 5, scaling/temperature, ba classifier (LSTM, Self-Attention, Self-Attention + pos), evaluation, application, error analysis
- results.csv: số liệu do notebook ghi ra (experiment, model, setting, split, metric, value)
- attention_heatmap.png: attention matrix của hai câu "The cat that I adopted yesterday is sleeping." / "The book that I bought yesterday was expensive." với bốn nguồn Q/K/V
- confusion_matrix.png: confusion matrix trên test của TF-IDF + LR, LSTM, Self-Attention, Self-Attention + pos
- Hình phụ: query_attention.png (Experiment 1 - 2), temperature.png, learning_curves.png, position_profile.png, sequence_cost.png (Experiment 5), application_attention.png
- error_analysis.md, reflection.md

## Cách chạy

```
pip install numpy scipy pandas scikit-learn torch gensim matplotlib pyarrow datasets jupyter
python attention_numpy.py        # unit test
```

Sau đó chạy toàn bộ self_attention.ipynb. Notebook dùng lại `../lab4/preprocessing.py` (load IMDb, tokenize, deduplicate, split) và đọc dữ liệu từ `../imdb/`. Chạy trên CPU mất khoảng 20 - 25 phút. Đặt biến môi trường `LAB6_QUICK=1` để chạy thử trên tập con (không ghi results.csv).

## Thiết lập

- Dữ liệu: IMDb, cùng split với LAB 04 - 05 (train 19,923 / validation 4,981 / test 24,678 sau khi bỏ duplicate).
- Experiment 1 - 3: Word2Vec Skip-gram (d = 100, window 5, min_count 5, 10 epoch) train trên train split, dùng làm embedding cố định.
- Classifier (cùng vocabulary 20,002, max_length 200, embedding 100 học từ đầu, batch 64, Adam 1e-3, 8 epoch, clip 1.0, giữ checkpoint có validation loss thấp nhất):
  - LSTM: giống LAB 05, dùng hidden state tại token thật cuối cùng.
  - Self-Attention: Q = XW_Q, K = XW_K, V = XW_V (100 → 128), H = softmax(QKᵀ/√128 + mask)V, attention pooling c = Σ β_t h_t với β = softmax(Hu/√128), u là query học được → Linear.
  - Self-Attention + pos: như trên, cộng thêm positional embedding học được vào X.
  - Self-attention viết tay theo đúng công thức của `attention_numpy.py`, không dùng `nn.MultiheadAttention`. Đã kiểm tra hai bản cho cùng kết quả khi dùng chung weight.
- Baseline: TF-IDF unigram + Logistic Regression (LAB 04).

## Kết quả chính

### Bài tính (kiểm tra bằng attention_numpy.py)

| Bài | Kết quả |
|---|---|
| Bài 1: scores | $s = [1, 2, 3]$ |
| Bài 2: softmax([1, 2, 3]) | $[0.090, 0.245, 0.665]$, tổng = 1 |
| Bài 3: context | $c = [1.5, 1.6]$ |
| Bài 4: toàn bộ | $s = [1, 0, 1]$, $\alpha = [0.422, 0.155, 0.422]$, $c = [1.267, 0.733]$ |
| Scaling ($d_k = 4$) | không scale $[0.867, 0.117, 0.016]$, chia 2 thì $[0.665, 0.245, 0.090]$ |
| Stability | naive softmax([1000, 1001, 1002]) = nan, bản ổn định = $[0.090, 0.245, 0.665]$ |

### Experiment 1 - 2 và temperature

- Query sentiment dồn 0.381 vào "good". Đổi sang query topic thì "movie" lên 0.236 còn "good" xuống 0.085: attention phụ thuộc vào query.
- Temperature 0.5 / 1 / 2: max weight 0.728 / 0.381 / 0.213, entropy 1.06 / 1.94 / 2.22 (tối đa 2.30).
- Không scaling thì max weight tăng từ 0.50 lên 0.97 khi $d_k$ từ 4 lên 1024, và gradient softmax giảm còn 0.046. Có scaling thì cả hai ổn định.

### Experiment 4 - LSTM vs attention (test, max_length 200)

| Model | Representation | Accuracy | Precision | Recall | F1 | Train time | Params |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | TF-IDF (toàn bộ review) | 0.890 | 0.892 | 0.889 | 0.890 | 5 s | 43,173 |
| LSTM | $h_T$ | 0.825 | 0.819 | 0.835 | 0.827 | 389 s | 2,118,089 |
| Self-Attention | $\sum_t \beta_t h_t$ | 0.837 | 0.819 | 0.867 | **0.842** | 270 s | 2,038,857 |
| Self-Attention + pos | $\sum_t \beta_t h_t$ | 0.833 | 0.832 | 0.836 | 0.834 | 300 s | 2,058,857 |

- Self-Attention không vị trí cho P(pos) **giống hệt nhau** với các câu cùng bag-of-words. Thêm positional embedding thì phân biệt được thứ tự.
- LSTM dồn 29.7% độ nhạy vào 20 token cuối. Attention phân bố đều theo vị trí, nhận ra từ khoá ở đầu review mà LSTM bỏ qua, và tốt hơn LSTM ở review dài hơn 400 token (0.768 so với 0.732).

### Experiment 5 - Chi phí

| T | Phần tử $T^2$ | A, batch 64, float32 | NumPy attention | PyTorch self-attention | PyTorch LSTM |
|---|---|---|---|---|---|
| 10 | 100 | 0.02 MB | 0.04 ms | 0.24 ms | 0.38 ms |
| 100 | 10,000 | 2.4 MB | 0.31 ms | 0.38 ms | 0.86 ms |
| 500 | 250,000 | 61 MB | 8.0 ms | 1.3 ms | 3.1 ms |
| 1,000 | 1,000,000 | 244 MB | 30.5 ms | 4.1 ms | 7.8 ms |
| 2,000 | 4,000,000 | 977 MB | 115 ms | 14.2 ms | 15.1 ms |
| 4,000 | 16,000,000 | 3.9 GB | 394 ms | 52.0 ms | 30.4 ms |

Độ dốc log-log: attention 1.78 - 1.88 (gần $T^2$), LSTM 1.08. Attention nhanh hơn LSTM khi T ≤ 1000, nhưng chậm hơn từ khoảng T = 2000.

### Attention và giải thích

- Pooling weight β gần như đều ($H/\ln n = 0.997$), không tương quan với occlusion (Spearman −0.03). Attention rollout βA tương quan 0.35, top-1 trùng occlusion 52.7%.
- Heatmap của hai model trained khác hẳn nhau và nhiều ô không có nghĩa ngôn ngữ: heatmap không chứng minh model hiểu ngữ pháp.
- Lỗi điển hình: không có thứ tự nên không xử lý được contrast ("would be bad, but it was actually excellent"), một từ mạnh lấn át cả câu, tự tin quá mức trên câu rất ngắn ("I like this movie." P = 0.000), rare word, nhãn nhiễu.

### Lưu ý về tái lập

Word2Vec (Experiment 1 - 3) train với 4 luồng nên số liệu có thể lệch nhẹ giữa các lần chạy (ví dụ weight của "good" là 0.381 ở lần chạy này, 0.395 ở lần trước). Ba classifier PyTorch dùng seed cố định nên cho cùng kết quả qua các lần chạy.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: attention_numpy.py, self_attention.ipynb, results.csv, các hình .png, bản nháp calculations.md, prediction.md, error_analysis.md, reflection.md
- Modified: không
- Verified: unit test trong attention_numpy.py (Bài 1 - 4, scaling, temperature, stability, bài self-attention, mask, so với torch.nn.functional.scaled_dot_product_attention), lớp SelfAttention của PyTorch khớp với bản NumPy khi dùng chung weight (sai số 1e-16), dự đoán khi có và không có padding trùng nhau, đối chiếu số liệu trong các file .md với output của notebook

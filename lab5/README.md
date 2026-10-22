# LAB 05 - Sequence Models: RNN, LSTM, GRU

## Files

- calculations.md: Bài 1 (một bước RNN), Bài 2 (information flow), Bài 3 (sequence order), vanishing/exploding gradient, LSTM gates, GRU
- prediction.md: prediction trước experiment và đối chiếu sau experiment
- rnn_cell.py: SimpleRNNCell tự implement bằng NumPy, unit test so với Bài 1 và torch.nn.RNNCell / torch.nn.RNN
- sequence_classification.ipynb: Experiment 1 - 6, baseline, evaluation, learning curves, error analysis, application
- results.csv: số liệu do notebook ghi ra (experiment, model, setting, split, metric, value)
- learning_curves.png: train / validation loss theo epoch của RNN, LSTM, GRU
- confusion_matrix.png: confusion matrix trên test của TF-IDF + LR, RNN, LSTM, GRU
- sequence_length_hist.png, sequence_length.png, gradient_by_position.png: hình phụ của Experiment 1, 5, 6
- error_analysis.md: confusion matrix, review đúng/sai, nguyên nhân lỗi
- reflection.md: so sánh với LAB 04, giới hạn của RNN, động lực dẫn tới attention, learning check

## Cách chạy

```
pip install numpy pandas scikit-learn torch matplotlib pyarrow datasets jupyter
python rnn_cell.py        # unit test
```

Sau đó chạy toàn bộ sequence_classification.ipynb. Notebook dùng lại `../lab4/preprocessing.py` (load IMDb, clean_text, tokenize, deduplicate, split) để có đúng split của LAB 04, và đọc dữ liệu từ `../imdb/`. Chạy trên CPU (8 thread) mất khoảng 35 phút. Đặt biến môi trường `LAB5_QUICK=1` để chạy thử trên tập con (không ghi results.csv).

## Dataset và split

Giống LAB 04: IMDb 25,000 train / 25,000 test, cân bằng 50/50, không có missing. Bỏ duplicate và 123 review test trùng với train, chia train/validation 80/20 (stratified, seed 42): train 19,923, validation 4,981, test 24,678.

## Thiết lập

- Tokenizer: bỏ thẻ HTML, lowercase, regex [a-z0-9]+('[a-z]+)? (giống LAB 02 - 04).
- Vocabulary: 20,000 từ phổ biến nhất trong train + PAD (id 0) + UNK (id 1).
- Padding ở cuối, truncation giữ max_length token đầu (head). max_length mặc định 200.
- Model: Embedding(20,002, 100, padding_idx = 0) → RNN / LSTM / GRU (1 layer, hidden 128, một chiều) → dropout 0.3 → Linear(128, 1). Classifier dùng hidden state tại token thật cuối cùng (tương đương pack_padded_sequence, đã kiểm tra).
- Train: BCE loss, Adam lr 1e-3, batch 64, gradient clipping 1.0, 8 epoch, giữ weight của epoch có validation loss thấp nhất. Mọi model dùng chung vocabulary, embedding dim, hidden size, batch, optimizer, lr, số epoch, seed, split.
- Baseline: TF-IDF unigram (min_df = 2, sublinear tf) + Logistic Regression của LAB 04, C chọn theo F1 trên validation.

## Kết quả chính (test, max_length = 200)

| Model | Representation | Params (recurrent) | Train time | Best epoch | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | TF-IDF unigram | 43,173 | 3.6 s | - | 0.890 | 0.892 | 0.889 | 0.890 |
| RNN | Embedding 100 | 2,029,769 (29,440) | 243 s | 5 / 8 | 0.750 | 0.752 | 0.749 | 0.751 |
| LSTM | Embedding 100 | 2,118,089 (117,760) | 294 s | 5 / 8 | 0.825 | 0.819 | 0.835 | 0.827 |
| GRU | Embedding 100 | 2,088,649 (88,320) | 776 s | 4 / 8 | 0.849 | 0.831 | 0.879 | 0.854 |

Vocabulary 20,002 cho sequence model (43,172 cho TF-IDF). Train time của sequence model là 8 epoch trên CPU.

### Sequence length (Experiment 2 + 5)

| Model | max_length | % token giữ lại | Train time | Activation / batch | Val loss | Accuracy | F1 |
|---|---|---|---|---|---|---|---|
| LSTM | 50 | 21.8% | 98 s | 26.9 MB | 0.503 | 0.737 | 0.742 |
| LSTM | 100 | 42.0% | 183 s | 53.4 MB | 0.445 | 0.781 | 0.773 |
| LSTM | 200 | 69.3% | 294 s | 106.3 MB | 0.416 | 0.825 | 0.827 |
| LSTM (giữ 100 token **cuối**) | 100 | 42.0% | 189 s | 53.4 MB | 0.387 | 0.838 | 0.838 |
| RNN | 50 / 100 / 200 | 21.8 / 42.0 / 69.3% | 73 / 142 / 243 s | 2.9 / 5.7 / 11.3 MB | 0.610 / 0.635 / 0.542 | 0.676 / 0.701 / 0.750 | 0.668 / 0.700 / 0.751 |

### Những điểm chính

- LSTM và GRU tốt hơn Simple RNN rõ rệt (F1 0.827 và 0.854 so với 0.751), nhưng **không model sequence nào vượt TF-IDF + LR** (0.890). Khoảng cách chủ yếu nằm ở review dài: với review ≤ 200 token, GRU đạt accuracy 0.875 - 0.882, gần bằng TF-IDF; với review > 400 token chỉ còn 0.758.
- Sequence model **có dùng thứ tự**: với các cặp câu cùng bag-of-words, TF-IDF cho cùng P(pos), còn RNN/LSTM/GRU cho kết quả khác nhau ("expected bad but it was great" 0.694 so với "expected great but it was bad" 0.328, LSTM). Nhưng chưa ổn định: "i loved it at first but now i hate it" vẫn bị đoán positive.
- Vanishing gradient thấy rõ: ‖∂z/∂e_t‖ của RNN ở token đầu chỉ bằng khoảng 10⁻¹⁰ lần token cuối, còn LSTM khoảng 0.03, GRU khoảng 0.3 (`gradient_by_position.png`). Khi câu quyết định nằm ở đầu và phía sau có khoảng 100 token, cả LSTM cũng không phân biệt được positive/negative.
- Vị trí thông tin quan trọng hơn độ dài: LSTM giữ 100 token cuối (F1 0.838) tốt hơn giữ 200 token đầu (0.827). Thời gian và bộ nhớ tăng tuyến tính theo max_length.
- Cả ba model overfit sau epoch 4 - 5 (validation loss tăng lại), notebook giữ checkpoint có validation loss thấp nhất.
- Lỗi điển hình: truncation (tỉ lệ lỗi 19.5% ở review bị cắt so với 12.1%), phần cuối lấn át phần đầu, contrast/negation chưa đảo được nghĩa, sarcasm và nhãn nhiễu.

## AI assistance statement

- Tool: Claude Code
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated: rnn_cell.py, sequence_classification.ipynb, results.csv, các hình .png, bản nháp calculations.md, prediction.md, error_analysis.md, reflection.md
- Modified: (sinh viên ghi các phần đã tự sửa)
- Verified: unit test trong rnn_cell.py (Bài 1, so với torch.nn.RNNCell / torch.nn.RNN với cùng weight, sai số 1e-16), kiểm tra hidden state tại token cuối khớp với pack_padded_sequence, số tham số tính tay khớp với PyTorch, đối chiếu số liệu trong các file .md với output của notebook

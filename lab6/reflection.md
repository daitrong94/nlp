# Reflection

## So sánh LAB 05 và LAB 06 (mục 28)

| Property | RNN / LSTM / GRU | Attention (full self-attention) | Quan sát trong lab |
|---|---|---|---|
| Sequence order | ✓ | ✓ **chỉ khi có positional information** | Self-Attention không vị trí cho P(pos) giống hệt nhau với mọi cặp câu cùng bag-of-words. Thêm positional embedding thì phân biệt được ("expected bad but it was great" 0.896 so với "... great but it was bad" 0.026) |
| Hidden state | ✓ | Không bắt buộc | Context $c = \sum_t \beta_t h_t$ là tổng có trọng số, không có trạng thái truyền qua thời gian |
| Sequential computation | ✓ | Không bắt buộc | Mọi hàng của $QK^\top$ tính cùng lúc |
| Direct token-to-token interaction | Limited | ✓ | adopted → cat 0.91 trong một bước, không qua chuỗi $h_1 \to \cdots \to h_T$ |
| Long-range interaction | Difficult | Easier | LSTM: 29.7% độ nhạy nằm ở 20 token cuối, từ khoá ở đầu bị bỏ qua. Attention: phân bố phẳng, từ khoá ở đầu và ở cuối cho cùng kết quả |
| Parallel computation | Limited | ✓ | 31.7 s/epoch so với 46.7 s/epoch. Với T = 1000, 4.1 ms so với 7.8 ms |
| Cost theo sequence length | Gần tuyến tính theo timestep | $O(T^2)$ | Độ dốc log-log 1.08 (LSTM) so với 1.78 - 1.88 (attention). Attention chậm hơn LSTM từ khoảng T = 2000 |
| Context representation | Recurrent state $h_T$ | Weighted combination $\sum_t \alpha_t v_t$ | |
| Test F1 trên IMDb (L = 200) | LSTM 0.827 | Self-Attention 0.842, + pos 0.834 | TF-IDF + LR vẫn đạt 0.890 |

### Attention giải quyết vấn đề gì của recurrent processing?

1. **Đường đi dài.** Ở RNN, thông tin của $x_1$ phải đi qua $T - 1$ phép biến đổi mới tới $h_T$, nên gradient tới token đầu co về 0 (LAB 05: tỉ lệ khoảng $10^{-10}$ với Simple RNN). Trong self-attention, mọi token nối tới context vector qua 1 - 2 phép nhân ma trận. Thí nghiệm cho thấy: phân bố độ nhạy theo vị trí của attention phẳng (khoảng 10% ở 20 token đầu, khoảng 10% ở 20 token cuối), còn LSTM chỉ 3.2% ở đầu và 29.7% ở cuối. Từ khoá đặt ở đầu một đoạn 100 token vẫn được attention nhận ra (logit separation 2.89), còn LSTM thì không (0.04).
2. **Bottleneck một vector.** RNN phải nén cả câu vào $h_T$. Attention giữ lại mọi $h_t$ và chọn bằng trọng số phụ thuộc vào query. Experiment 2: cùng một câu, Query A dồn 0.381 vào "good", Query B dồn 0.236 vào "movie".
3. **Tính tuần tự.** $QK^\top$ và $AV$ là phép nhân ma trận, chạy song song được. Với sequence không quá dài, attention train nhanh hơn LSTM khoảng 1.4 lần.
4. Trên IMDb, attention tốt hơn LSTM ở review dài (accuracy 0.768 so với 0.732 với review > 400 token) và bằng LSTM ở review ngắn.

### Attention tạo ra vấn đề mới nào?

1. **Chi phí $O(T^2)$** (mục 29 - "attention không miễn phí"). T = 1000 cho 1 triệu phần tử, T = 10,000 cho 100 triệu. Với batch 64, float32, một head, ma trận A ở T = 4000 đã chiếm 3.9 GB. Thời gian đo được tăng gần bình phương (độ dốc 1.88), và vượt LSTM từ khoảng T = 2000.
2. **Mất thứ tự.** Self-attention là permutation-equivariant: không có thông tin vị trí thì model chỉ là bag-of-words phi tuyến. Thí nghiệm xác nhận điều này một cách chính xác, nên cần positional encoding (LAB 07).
3. **Pha loãng khi pooling.** Pooling weight β học được gần như đều ($H(\beta)/\ln n = 0.997$). Một câu quyết định giữa 150 token trung tính chỉ còn logit separation 2.12, so với 5.69 khi chỉ có khoảng 37 token.
4. **"Winner-take-all" và tự tin quá mức.** Khi nhiều query cùng chọn một key mạnh, một từ cảm xúc quyết định cả câu: "The cast is wonderful, but the story is boring and the ending is awful" ra P = 0.997, với Δz(wonderful) = +15.97. Trên câu rất ngắn, model cho P = 0.000 với "I like this movie."
5. **Overfit nhanh.** Validation loss tăng từ epoch 3. Self-Attention + pos có train loss 0.005 ở epoch 8.
6. **Diễn giải dễ sai.** Heatmap trông "có cấu trúc" (adopted → cat 0.91, is → cat 0.96) nhưng hai model với accuracy gần nhau cho hai pattern khác hẳn, và có nhiều ô không có nghĩa ngôn ngữ (yesterday → the 0.97). β gần đều không tương quan với occlusion (Spearman −0.03). Rollout βA tốt hơn (0.35) nhưng vẫn chỉ trùng top-1 trong khoảng một nửa số review.

Kết luận: không nên hiểu "attention tốt hơn RNN nên luôn thay thế RNN". Attention đổi vấn đề đường đi dài lấy chi phí bình phương, và cần thêm vị trí cùng nhiều cơ chế khác mới thành một architecture hoàn chỉnh.

## Attention visualization không phải là reasoning

- Word2Vec với $W = I$: mỗi token chủ yếu chú ý tới chính nó, vì $q_i \cdot k_i = \lVert x_i \rVert^2$. W ngẫu nhiên: gần đều. Cả hai không có tham số nào được học.
- Model được train: pattern sắc nét nhưng phụ thuộc vào model, và phần lớn không có diễn giải ngôn ngữ.
- Self-Attention không vị trí cho hàng "adopted" giống nhau với mọi thứ tự từ. Vì vậy adopted → cat không thể là quan hệ động từ - tân ngữ dựa trên cấu trúc câu.

Heatmap là bằng chứng về **phân phối attention**. Muốn nói về reasoning phải có thí nghiệm can thiệp (occlusion, đổi thứ tự, đổi từ) và đo trên nhiều câu.

## Từ LAB 06 sang LAB 07

Lab này để lại hai câu hỏi:

1. **Làm sao thực hiện attention cho cả sequence một cách hiệu quả, và có thứ tự?** Cần positional encoding (thí nghiệm "+ pos" đã cho thấy tác dụng), viết attention dưới dạng ma trận cho mọi query cùng lúc (đã làm: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$), và chấp nhận hoặc giảm chi phí $O(T^2)$.
2. **Làm sao xây một architecture hoàn chỉnh?** Một head chỉ học được một kiểu quan hệ (ở đây chủ yếu là "chép từ cảm xúc mạnh"). Multi-head attention cho nhiều kiểu quan hệ song song. Residual, layer norm và feed-forward giúp xếp chồng nhiều layer. Ghép lại thành Transformer block, tức là nội dung của LAB 07.

## Individual learning check

- **Attention score được tính như thế nào?** $s_i = q^\top k_i$ (dot product giữa query và từng key), hoặc chia thêm $\sqrt{d_k}$: $s_i = q^\top k_i / \sqrt{d_k}$. Score lớn khi q và k cùng hướng và có độ lớn lớn. Ví dụ Bài 1: $q = [1, 2]$, $K = [[1,0],[0,1],[1,1]]$ cho $s = [1, 2, 3]$.
- **Tại sao phải dùng softmax?** Để biến các score (âm, dương, không giới hạn) thành trọng số dương có tổng bằng 1, nên context là một trung bình có trọng số của các value. Softmax giữ thứ tự, khuếch đại chênh lệch ($\alpha_i/\alpha_j = e^{s_i - s_j}$), và khả vi nên train được bằng gradient. Ví dụ $\operatorname{softmax}([1, 2, 3]) = [0.090, 0.245, 0.665]$.
- **Query, Key và Value có vai trò gì?** Query là "đang tìm gì" (của token hiện tại hoặc của task). Key là "mỗi token có gì để được tìm thấy", dùng để so với query. Value là "thông tin thật sự được lấy ra" khi token được chọn. Tách K và V cho phép tiêu chí chọn khác với nội dung lấy ra. Trong self-attention, cả ba là ba phép chiếu khác nhau của cùng X: $Q = XW_Q$, $K = XW_K$, $V = XW_V$.
- **Context vector được tính như thế nào?** $c = \sum_i \alpha_i v_i$, với $\alpha = \operatorname{softmax}(s)$. Ví dụ Bài 3: $0.1[1,0] + 0.2[0,1] + 0.7[2,2] = [1.5, 1.6]$. Bài 4: $c = [1.267, 0.733]$.
- **Tại sao scaled dot-product attention chia cho $\sqrt{d_k}$?** Nếu thành phần của q, k có phương sai 1 thì $\operatorname{Var}(q^\top k) = d_k$, nên score lớn dần theo số chiều và softmax bão hoà gần one-hot, gradient gần 0. Đo được: không scaling, với $d_k = 1024$ thì max weight là 0.97 và gradient softmax là 0.046. Có scaling thì luôn khoảng 0.32 và 0.35. Lưu ý quy tắc này giả định thành phần có phương sai 1: với Word2Vec (độ lệch chuẩn 0.24), chia $\sqrt{100}$ làm attention gần như đều.
- **Self-attention khác attention thông thường ở đâu?** Ở attention thông thường, query có thể đến từ nguồn khác key/value (ví dụ query sentiment giả lập ở Experiment 1, hay query u của pooling). Trong self-attention, Q, K, V đều tạo từ cùng một sequence, nên mỗi token vừa hỏi vừa được hỏi bởi các token khác, và output là biểu diễn mới của từng token theo context của câu (ma trận $T \times T$).
- **Nếu sequence dài gấp đôi thì attention matrix thay đổi như thế nào?** Từ $T \times T$ thành $2T \times 2T$, tức gấp 4 lần số phần tử, bộ nhớ và (gần như) thời gian. Ví dụ 1000 token cho 1 triệu phần tử, 2000 token cho 4 triệu. Đo được 30.5 ms → 115 ms (×3.8) với NumPy.
- **Attention có chứng minh model đang reasoning không?** Không. Attention chỉ là trọng số trộn các value. Trong lab: hai model gần bằng nhau về accuracy có heatmap khác hẳn; β gần đều không tương quan với ảnh hưởng thật của token (Spearman −0.03); model không vị trí cho adopted → cat 0.91 bất kể thứ tự từ. Muốn nói về reasoning phải kiểm tra bằng can thiệp.
- **Tại sao attention có thể xử lý long-range interaction tốt hơn recurrent state?** Vì mọi cặp token tương tác trực tiếp qua $q_i^\top k_j$ trong một bước. Thông tin và gradient không phải đi qua chuỗi $T - t$ phép biến đổi, và không có tích Jacobian dài để co về 0. Trong lab, độ nhạy theo vị trí của attention phẳng, còn của LSTM dồn về cuối, và từ khoá ở đầu review vẫn ảnh hưởng tới prediction của attention. Đổi lại phải trả chi phí $O(T^2)$ và phải thêm thông tin vị trí.

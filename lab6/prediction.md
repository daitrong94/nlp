# Prediction

Prediction, Reason và Confidence được ghi trước khi chạy code. Dòng "Sau experiment" ghi sau khi có kết quả.

## Prediction 1 - Key có dot product lớn hơn các key khác thì attention weight của nó thế nào?

- Prediction: Weight của key đó lớn nhất. Nó lớn hơn các key khác theo tỉ lệ $e^{\Delta s}$, với $\Delta s$ là chênh lệch score, chứ không theo tỉ lệ của bản thân score.
- Reason: $\alpha_i = e^{s_i} / \sum_j e^{s_j}$. Hàm mũ đồng biến, mẫu số giống nhau cho mọi $i$, nên thứ tự của $\alpha$ giống thứ tự của $s$. Tỉ số giữa hai weight là $\alpha_i / \alpha_j = e^{s_i - s_j}$. Ví dụ chênh 1 đơn vị score thì weight gấp $e \approx 2.7$ lần.
- Confidence: 95%.
- Sau experiment: Đúng. Ở Experiment 1 (Query A), "good" có score cao nhất (4.17) và nhận weight lớn nhất (0.381). "slow" có score 3.18 và nhận 0.141. Tỉ số $0.381 / 0.141 = 2.70 \approx e^{4.17 - 3.18} = e^{0.99}$, đúng như công thức.

## Prediction 2 - Tất cả scores bằng nhau, $s = [1, 1, 1, 1]$

- Prediction: Attention đều, mỗi token nhận $1/4 = 0.25$. Context vector là trung bình cộng các value, giống Average Word2Vec ở LAB 04. Entropy đạt cực đại $\ln 4 \approx 1.386$.
- Reason: $e^{1}/(4e^{1}) = 1/4$. Softmax chỉ phụ thuộc vào chênh lệch giữa các score: cộng cùng một hằng số vào mọi score không đổi kết quả.
- Confidence: 95%.
- Sau experiment: Đúng. `softmax([1, 1, 1, 1]) = [0.25, 0.25, 0.25, 0.25]`, entropy $= \ln 4$ (kiểm tra trong `attention_numpy.py` và notebook).

## Prediction 3 - Một score lớn hơn rất nhiều, $s = [1, 1, 10]$

- Prediction: Gần như toàn bộ attention dồn vào token thứ ba, khoảng 0.9998. Hai token còn lại mỗi token khoảng $10^{-4}$. Context vector gần như bằng $v_3$, tức attention hoạt động như một phép chọn cứng (gần argmax).
- Reason: $\alpha_3 = 1/(1 + 2e^{-9}) \approx 1 - 2.5 \cdot 10^{-4}$. Khi softmax bão hoà như vậy, gradient theo các score còn lại gần bằng 0, nên đây cũng là lý do phải chia cho $\sqrt{d_k}$.
- Confidence: 90%.
- Sau experiment: Đúng. `softmax([1, 1, 10]) = [0.000123, 0.000123, 0.999753]`. Thí nghiệm scaling cũng cho thấy hệ quả về gradient: không chia $\sqrt{d_k}$ thì với $d_k = 1024$, max weight trung bình là 0.97 và $\lVert \partial\alpha / \partial s \rVert$ chỉ còn 0.046, so với 0.35 khi có scaling.

## Prediction 4 - Sequence tăng từ 10 lên 1000 tokens, chi phí attention thay đổi thế nào?

- Prediction: Số phần tử của attention matrix tăng $100^2 = 10{,}000$ lần (từ 100 lên 1,000,000). Thời gian tính $QK^\top$ và $AV$ cũng tăng khoảng 10,000 lần nếu $d$ cố định. Trong thực tế, với T nhỏ, chi phí cố định (gọi hàm, cấp phát bộ nhớ) chiếm phần lớn nên tỉ lệ đo được sẽ thấp hơn. LSTM chỉ tăng khoảng 100 lần (tuyến tính), nhưng không song song được theo T.
- Reason: Self-attention tính score cho mọi cặp $(i, j)$, có $T^2$ cặp, mỗi cặp tốn $O(d)$. Tổng là $O(T^2 d)$ thời gian và $O(T^2)$ bộ nhớ cho ma trận A.
- Confidence: 80% cho tỉ lệ số phần tử (đây là phép tính), 60% cho tỉ lệ thời gian đo được.
- Sau experiment: Đúng cả hai ý. Số phần tử tăng từ 100 lên 1,000,000 (×10,000). Thời gian NumPy attention đo được tăng từ 0.041 ms lên 30.5 ms, tức ×745, thấp hơn ×10,000 vì với T nhỏ chi phí cố định chiếm phần lớn. Với T lớn, độ dốc log-log là 1.88 (gần 2), còn LSTM là 1.08. Điều mình không dự đoán được: với T ≤ 1000, self-attention trong PyTorch lại **nhanh hơn** LSTM (4.1 ms so với 7.8 ms ở T = 1000) nhờ tính song song. Hai đường chỉ cắt nhau ở khoảng T = 2000.

## Prediction 5 - Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?

- Prediction: Không.
- Reason: Attention weight được học để giảm loss, không được học để giải thích cho con người. Một token có weight cao chưa chắc quyết định prediction, vì value của nó có thể gần trung tính, hoặc thông tin đã được trộn từ các token khác qua self-attention. Ngược lại, một token có weight thấp vẫn có thể ảnh hưởng lớn nếu value của nó lớn. Ngoài ra, model có thể dựa vào shortcut của dataset (ví dụ điểm "7/10", tên diễn viên) thay vì từ cảm xúc. Mình dự đoán thứ hạng theo attention và thứ hạng theo occlusion (bỏ token rồi xem prediction đổi bao nhiêu) chỉ tương quan vừa phải.
- Confidence: 80%.
- Sau experiment: Đúng. Trên 300 review, Spearman giữa pooling weight β và occlusion $|\Delta z|$ là −0.03, giữa attention rollout βA và $|\Delta z|$ là 0.35. Token có βA cao nhất trùng với token quan trọng nhất theo occlusion trong 52.7% review. Có những trường hợp model dựa vào token mà con người không coi là quan trọng: review 250 có βA và occlusion cùng xếp "prospective" đứng đầu; review 18286 có βA cao nhất ở "is" và "the". Attention cho biết model dựa vào đâu, không cho biết đó có phải là chỗ con người cho là quan trọng.

## Thêm - Self-attention classifier hay LSTM tốt hơn trên IMDb (cùng max_length 200)?

- Prediction: Gần bằng nhau, chênh khoảng ±3 điểm F1. Self-attention train nhanh hơn trên CPU vì tính song song mọi vị trí. Self-attention **không có positional encoding** sẽ không phân biệt được thứ tự: hoán vị các token cho cùng một output, giống bag-of-words. Ngược lại, nó sẽ không bị ảnh hưởng bởi vị trí của từ khoá như LSTM ở LAB 05.
- Reason: Self-attention chỉ dùng $X W_Q$, $X W_K$, $X W_V$. Hoán vị các hàng của X thì các hàng của output bị hoán vị theo, và pooling bằng tổng có trọng số không phụ thuộc thứ tự. Mọi token có đường đi dài 1 tới context vector.
- Confidence: 65% cho ý gần bằng nhau, 95% cho ý không phân biệt thứ tự (đây là tính chất toán học).
- Sau experiment: Đúng phần lớn. F1: Self-Attention 0.842, Self-Attention + pos 0.834, LSTM 0.827, đều nằm trong khoảng ±3 điểm, và attention còn cao hơn LSTM. Thời gian mỗi epoch 31.7 s so với 46.7 s. Self-Attention không có vị trí cho P(pos) **giống hệt nhau** với mọi cặp câu cùng bag-of-words, và cho cùng separation dù từ khoá ở đầu, giữa hay cuối. Điều mình không dự đoán được: (1) separation vẫn giảm mạnh khi đoạn filler dài ra (logit 5.69 → 2.12), vì pooling weight gần đều nên context bị pha loãng; (2) model attention rất tự tin và sai trên câu ngắn ("I like this movie." P(pos) = 0.000), trong khi LSTM đúng.

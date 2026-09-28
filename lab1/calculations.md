# Calculations — Part B

> Hoàn thành phần này bằng tay, không dùng AI/code, trước khi kiểm chứng.

## Exercise 1 — Count Vector

Vocabulary: `[cat, dog, eats, fish, likes]`

- D1 = "cat eats fish" → [1, 0, 1, 1, 0]
- D2 = "dog eats fish" → [0, 1, 1, 1, 0]
- D3 = "cat likes fish" → [1, 0, 0, 1, 1]

## Exercise 2 — TF

D1 = "cat eats fish" (3 token, mỗi từ xuất hiện 1 lần)

- tf(cat, D1) = 1/3
- tf(eats, D1) = 1/3
- tf(fish, D1) = 1/3
- Kiểm tra: 1/3 + 1/3 + 1/3 = 1 ✓

## Exercise 3 — IDF

N = 3, df(cat)=2, df(dog)=1, df(eats)=2, df(fish)=3, df(likes)=1 (dùng ln)

- idf(cat) = ln(3/2) ≈ 0.405
- idf(dog) = ln(3/1) ≈ 1.099
- idf(eats) = ln(3/2) ≈ 0.405
- idf(fish) = ln(3/3) = 0
- idf(likes) = ln(3/1) ≈ 1.099

Term có IDF thấp nhất: **fish**, vì df(fish) = 3 — xuất hiện ở mọi document nên tỉ số N/df = 1, log(1) = 0, không còn tính phân biệt giữa các document.

## Exercise 4 — TF-IDF

D1 = "cat eats fish"

- tfidf(cat, D1) = 1/3 × 0.405 ≈ 0.135
- tfidf(eats, D1) = 1/3 × 0.405 ≈ 0.135
- tfidf(fish, D1) = 1/3 × 0 = 0

Tại sao fish xuất hiện trong mọi document nhưng TF-IDF bằng 0: vì idf(fish) = 0, và tfidf = tf × idf - tích luôn bằng 0 bất kể tf lớn hay nhỏ. Từ xuất hiện ở mọi document không giúp phân biệt document này với document khác, nên TF-IDF triệt tiêu nó.

## Exercise 5 — Cosine Similarity

x = [1, 1, 1], y = [1, 1, 0]

- x·y = 1×1 + 1×1 + 1×0 = 2
- ‖x‖ = √3 ≈ 1.732, ‖y‖ = √2 ≈ 1.414
- cos(x, y) = 2 / (1.732 × 1.414) ≈ **0.816**

Giải thích: 2/3 chỉ là tỉ lệ đếm thô (2 điểm chung / 3 chiều tổng cộng), còn cosine chia cho **độ dài (norm)** của từng vector chứ không chia cho tổng số chiều. Vì x có thêm 1 chiều khác 0 mà y không có, ‖x‖ dài hơn ‖y‖, khiến mẫu số (‖x‖‖y‖ ≈ 2.449) khác với cách tính thô (3), nên kết quả cosine (≈0.816) khác 2/3 (≈0.667).

## Exercise 6 — Prediction (không dùng code)

```
D1 = "medical image classification"
D2 = "medical image analysis"
D3 = "natural language processing"
Q  = "medical image classification"
```

1. Document similarity cao nhất: **D1** — trùng hoàn toàn với Q (3/3 từ khớp).
2. Document similarity thấp nhất: **D3** — không chia sẻ từ nào với Q.
3. Term có thể IDF thấp: **"medical"** và **"image"** — xuất hiện ở 2/3 document (D1, D2), df lớn nên idf nhỏ hơn các từ chỉ xuất hiện 1 lần như "classification", "analysis", "natural", "language", "processing".
4. Nếu bỏ IDF, chỉ dùng count vector: ranking tổng thể không đổi nhiều (D1 vẫn cao nhất vì trùng toàn bộ Q), nhưng khoảng cách giữa D1 và D2 sẽ thu hẹp lại — vì "classification" (từ đặc trưng chỉ có ở D1, so với "medical"/"image" dùng chung) mất đi trọng số phân biệt mà lẽ ra IDF sẽ tạo ra.

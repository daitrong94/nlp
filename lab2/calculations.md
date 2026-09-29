# Calculations - tính toán bằng tay

Corpus dùng cho Bài 1 - 4:

the cat eats fish
the cat likes fish
the dog eats meat

## Bài 1 - Unigram

1. Vocabulary: {the, cat, eats, fish, likes, dog, meat}, V = 7.

2. Tổng số token: N = 4 + 4 + 4 = 12.

3. Count: the = 3, cat = 2, eats = 2, fish = 2, likes = 1, dog = 1, meat = 1.

- P(the) = 3/12 = 0.25
- P(cat) = 2/12 = 1/6 ≈ 0.1667
- P(fish) = 2/12 = 1/6 ≈ 0.1667
- P(dog) = 1/12 ≈ 0.0833

4. Σ P(w) = (3 + 2 + 2 + 2 + 1 + 1 + 1)/12 = 12/12 = 1.

## Bài 2 - Bigram

C(the) = 3, C(cat) = 2.

- P(cat|the) = C(the cat)/C(the) = 2/3 ≈ 0.6667
- P(dog|the) = C(the dog)/C(the) = 1/3 ≈ 0.3333
- P(eats|cat) = C(cat eats)/C(cat) = 1/2 = 0.5
- P(likes|cat) = C(cat likes)/C(cat) = 1/2 = 0.5

Tại sao tổng xác suất các từ đứng sau the phải bằng 1: mỗi lần the xuất hiện làm context thì luôn có đúng một từ theo sau (kể cả token kết câu EOS nếu the đứng cuối câu), nên Σ_w C(the, w) = C(the). Do đó Σ_w P(w|the) = Σ_w C(the, w)/C(the) = 1. Ở đây P(cat|the) + P(dog|the) = 2/3 + 1/3 = 1.

## Bài 3 - Xác suất câu

P(the cat eats fish) = P(the) P(cat|the) P(eats|cat) P(fish|eats)

Với P(fish|eats) = C(eats fish)/C(eats) = 1/2:

P = 1/4 × 2/3 × 1/2 × 1/2 = 1/24 ≈ 0.0417

Nếu thêm một từ vào câu, xác suất cả câu không thể tăng: xác suất mới bằng xác suất cũ nhân thêm một thừa số P(w_new|w_prev) ≤ 1, nên chỉ giảm hoặc giữ nguyên (khi thừa số bằng 1). Vì vậy câu dài hơn bị phạt, không so sánh trực tiếp xác suất của các câu khác độ dài mà phải chuẩn hóa theo số token (log-prob trung bình hoặc perplexity).

## Bài 4 - Sentence ranking

- P(S1) = P(the) P(cat|the) P(eats|cat) P(fish|eats) = 1/4 × 2/3 × 1/2 × 1/2 = 1/24
- P(S2) = P(the) P(dog|the) P(eats|dog) P(fish|eats) = 1/4 × 1/3 × 1 × 1/2 = 1/24

Dự đoán: hai câu có xác suất bằng nhau (1/24 ≈ 0.0417). P(cat|the) gấp đôi P(dog|the) nhưng P(eats|dog) = 1 lại gấp đôi P(eats|cat) = 1/2 nên hai thừa số bù trừ nhau.

## Mục 9 - Suy luận trước smoothing

Corpus: I like NLP / I like AI / I study NLP

1. C(study AI) = 0.

2. P_MLE(AI|study) = C(study AI)/C(study) = 0/1 = 0.

3. Mọi câu chứa bigram study AI có xác suất bằng 0 (log-prob = -∞), perplexity trên dữ liệu chứa câu đó là vô cùng.

4. Không. Xác suất 0 chỉ phản ánh việc bigram này chưa xuất hiện trong corpus rất nhỏ (study chỉ xuất hiện 1 lần), không phải câu I study AI không thể xảy ra. Không quan sát thấy khác với xác suất thực sự bằng 0.

## Mục 11 - Laplace smoothing

C(cat) = 10, V = 5, P_Laplace(eats|cat) = (C(cat eats) + 1)/(C(cat) + V)

- C(cat eats) = 0: P_Laplace(eats|cat) = (0 + 1)/(10 + 5) = 1/15 ≈ 0.0667 (MLE: 0)
- C(cat eats) = 3: P_Laplace(eats|cat) = (3 + 1)/(10 + 5) = 4/15 ≈ 0.2667 (MLE: 3/10 = 0.3)

Smoothing thay đổi xác suất các bigram khác: mẫu số tăng từ 10 lên 15 cho mọi bigram bắt đầu bằng cat, nên một bigram có count c đổi từ c/10 thành (c + 1)/15. So sánh 15c với 10c + 10: bigram có c = 0 hoặc 1 tăng xác suất, c = 2 giữ nguyên (0.2), c > 2 bị giảm (ví dụ c = 3: 0.3 → 0.2667). Tức là khối xác suất được lấy bớt từ các bigram xuất hiện nhiều để chia cho các bigram hiếm và chưa gặp, tổng vẫn bằng 1.

## Mục 18 - Perplexity

P(w1) = 0.5, P(w2|w1) = 0.25, P(w3|w2) = 0.5, N = 3

- P(W) = 0.5 × 0.25 × 0.5 = 0.0625
- PP(W) = 0.0625^(-1/3) = 16^(1/3) ≈ 2.520

Khi P(w2|w1) = 0.1:

- P(W) = 0.5 × 0.1 × 0.5 = 0.025
- PP(W) = 0.025^(-1/3) = 40^(1/3) ≈ 3.420

Vì sao một xác suất nhỏ làm perplexity thay đổi đáng kể: P(W) là tích nên một thừa số giảm 2.5 lần kéo cả P(W) giảm 2.5 lần, perplexity là trung bình nhân của 1/P nên tăng (2.5)^(1/3) ≈ 1.36 lần (từ 2.52 lên 3.42) dù hai thừa số còn lại không đổi. Trong không gian log, log P → -∞ khi P → 0, nên một token có xác suất rất nhỏ có thể chi phối cả tổng; nếu P = 0 thì perplexity = ∞.

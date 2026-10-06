# Calculations

## Bài 1 - Co-occurrence (k = 1, bỏ "the")

Các cặp láng giềng: cat–eats, eats–fish, dog–eats, eats–fish, cat–likes, likes–milk, dog–likes, likes–meat.

| Word | cat | dog | eats | likes | fish | milk | meat |
|---|---|---|---|---|---|---|---|
| cat | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| dog | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| eats | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| likes | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

cat và dog có vector giống hệt nhau, nên cos = 1. cos(eats, likes) = 2/(√6 · 2) ≈ 0.408.

## Bài 2 - Similarity

x · y = 2 + 8 + 2 = 12, |x| = √6, |y| = 2√6, nên cos = 12/12 = 1.

y = 2x: hai vector cùng hướng, chỉ khác độ dài. Cosine bỏ qua độ lớn, mà độ lớn của vector đếm chủ yếu phản ánh tần suất. Vì vậy cos = 1 nghĩa là hai từ có cùng phân phối context, dù tần suất khác nhau.

## Bài 3 - Semantic similarity

Dự đoán: physician gần doctor hơn banana.

- cos(doctor, physician) = 1.14/(√1.14 · √1.17) ≈ 0.987
- cos(doctor, banana) = −0.14/(√1.14 · √0.86) ≈ −0.141

## Bài 4 - Sparse vs dense

1. Sparse: vector 10,000 chiều, chỉ 30 thành phần khác 0.
2. Dense: embedding 300 chiều.
3. Với vector sparse, hai từ đồng nghĩa có thể không chung context word nào, nên cos = 0. Dense embedding gộp các context tương tự vào cùng các chiều nên khái quát tốt hơn, và cũng gọn hơn khi tính toán.
4. Không phải lúc nào dense cũng tốt hơn. Sparse dễ giải thích, khớp chính xác từ khoá (tên riêng, mã sản phẩm), không cần train, và ở đây chỉ cần lưu 30 số. Dense cần nhiều dữ liệu, còn xếp các từ trái nghĩa gần nhau (hot/cold).

## Mục 12 - |V| = 100,000

Ma trận có 10^10 ô, tức khoảng 80 GB nếu lưu dense float64. Embedding 300 chiều cho cùng vocabulary chỉ khoảng 120 MB.

## Mục 16 - CBOW vs Skip-gram ("the cat eats fish", window = 1)

CBOW (context → target): [cat] → the, [the, eats] → cat, [cat, fish] → eats, [eats] → fish.

Skip-gram (target → context): the→cat, cat→the, cat→eats, eats→cat, eats→fish, fish→eats.

CBOW lấy context làm input để đoán từ giữa (4 example). Skip-gram lấy từ giữa để đoán từng context word (6 cặp), nên chậm hơn nhưng tốt hơn với từ hiếm.

## Mục 23 - Analogy

king − man + woman = [8 − 5 + 5, 2 − 1 + 3, 7 − 5 + 5] = [8, 4, 7]

woman − man = [0, 2, 0], tức chiều 2 là hướng giới tính. Kết quả giữ phần "hoàng gia" của king và đổi giới tính, nên gần với queen. Đây chỉ là một pattern hình học trong vector space, không chứng minh model hiểu quan hệ đó.

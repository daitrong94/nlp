# Calculations

## Bài 1 - Start / End

Context đã tokenize (đánh số từ 0): `[The, capital, of, France, is, Paris, .]`

| Answer | start | end |
|---|---|---|
| Paris | 5 | 5 |
| capital of France | 1 | 3 |

Answer một token thì start = end. Answer nhiều token thì start là token đầu, end là token cuối (tính cả token đó).

## Bài 2 - Start / End loss

Start đúng ở token 5, $P_{start}(5) = 0.60$:

$$
L_{start} = -\ln 0.60 = 0.511
$$

Giả sử $P_{end}(5) = 0.70$ thì $L_{end} = -\ln 0.70 = 0.357$, và:

$$
L = \frac{L_{start} + L_{end}}{2} = \frac{0.511 + 0.357}{2} = 0.434
$$

**Vì sao cần hai prediction?** Một span được xác định bởi hai vị trí. Nếu chỉ dự đoán start thì không biết answer dài bao nhiêu. Dùng hai phân phối $P_{start}$ và $P_{end}$, mỗi phân phối có $T$ lựa chọn, đơn giản hơn nhiều so với chấm điểm trực tiếp cả $T^2$ cặp (start, end).

## Bài 3 - Character span → token span

Context: "The capital of France is Paris."

1. Ký tự đầu của "Paris" có index **25**.
2. Ký tự cuối có index **29**.
3. Character span theo kiểu Python là `[25, 30)`, tức `context[25:30] = "Paris"`.
4. Sau khi tokenize cặp (question, context) bằng `distilbert-base-cased`:

| Token idx | 0 | 1 - 7 | 8 | 9 | 10 | 11 | 12 | 13 | **14** | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Token | [CLS] | What ... ? | [SEP] | The | capital | of | France | is | **Paris** | . | [SEP] |
| Offset | (0,0) | (câu hỏi) | (0,0) | (0,3) | (4,11) | (12,14) | (15,21) | (22,24) | **(25,30)** | (30,31) | (0,0) |
| sequence_id | - | 0 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - |

5. **Start token = 14**: offset (25, 30) chứa ký tự 25.
6. **End token = 14**: offset (25, 30) chứa ký tự cuối 29.

Ba loại index khác nhau: character index (25), word index ("Paris" là từ thứ 5 trong context), và token index (14, vì câu hỏi cùng [CLS] và [SEP] đứng trước context). Offset mapping là cầu nối giữa character index và token index.

## Bài 4 - Token-level F1

Gold "Paris France", Prediction "France": overlap = 1 token.

$$
P = \frac{1}{1} = 1, \quad R = \frac{1}{2} = 0.5, \quad F1 = \frac{2 \cdot 1 \cdot 0.5}{1 + 0.5} = 0.667
$$

EM = 0 vì hai chuỗi không giống hệt nhau, nhưng F1 = 0.667 vì prediction đúng một phần.

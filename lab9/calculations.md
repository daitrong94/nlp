# Calculations

## Bài 1 - BIO

Câu "Elon Musk founded SpaceX in California." với Elon Musk là PER, SpaceX là ORG, California là LOC:

| Word | Elon | Musk | founded | SpaceX | in | California | . |
|---|---|---|---|---|---|---|---|
| BIO | B-PER | I-PER | O | B-ORG | O | B-LOC | O |

- Từ đầu tiên của mỗi entity nhận `B-`, các từ tiếp theo trong cùng entity nhận `I-`, từ ngoài entity nhận `O`.
- SpaceX và California chỉ có một từ nên chỉ có `B-`, không có `I-`.

## Bài 2 - Entity boundary

| Word | University | of | California | Berkeley |
|---|---|---|---|---|
| BIO | B-ORG | I-ORG | I-ORG | I-ORG |

**Vì sao không gán tất cả là ORG?**

- Nếu chỉ dùng nhãn loại (`ORG ORG ORG ORG`), ta biết các từ này thuộc ORG nhưng không biết chúng là **một** entity 4 từ, hay là hai entity "University of California" và "Berkeley" đứng cạnh nhau.
- Hai entity cùng loại đứng liền nhau là chuyện có thật. Ví dụ "... gave [Microsoft] [Google] shares": BIO ghi là `B-ORG B-ORG`, còn khi chỉ dùng nhãn loại thì hai entity dính thành một.
- Entity-level metric so sánh theo span (start, end). Vì vậy nhãn phải mã hóa được ranh giới, chứ không chỉ loại entity.

## Bài 3 - Shape

$B = 8$, $T = 128$, $d = 768$, $C = 9$:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| Hidden states $H$ | $8 \times 128 \times 768$ | Mỗi token có một vector contextual $d$ chiều |
| Head $W$, $b$ | $768 \times 9$, $9$ | Dùng chung cho mọi token: $768 \cdot 9 + 9 = 6921$ tham số |
| Logits $Z = HW + b$ | $8 \times 128 \times 9$ | Mỗi token có 9 logit, một logit cho mỗi nhãn BIO |
| Labels $Y$ | $8 \times 128$ | Một id nhãn cho mỗi token, `-100` (IGNORE) ở vị trí không tính loss |

So với LAB 08: sequence classification chỉ lấy $H[:, 0]$ ($8 \times 768$) nên logits là $8 \times C$. Token classification áp head lên **mọi** vị trí.

DistilBERT-base-cased dùng trong lab có đúng $d = 768$, nên các shape này cũng là shape thật khi chạy (với $T$ là độ dài của batch).

## Label alignment với tokenizer thật (Strategy A)

Tokenizer `distilbert-base-cased`:

| Token | [CLS] | El | ##on | Mu | ##sk | founded | Space | ##X | in | California | . | [SEP] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| word_id | - | 0 | 0 | 1 | 1 | 2 | 3 | 3 | 4 | 5 | 6 | - |
| label | IGN | B-PER | IGN | I-PER | IGN | O | B-ORG | IGN | O | B-LOC | O | IGN |

- 7 word thành 10 token (không tính [CLS] và [SEP]). Ba tên riêng "Elon", "Musk", "SpaceX" bị tách thành subword, trong khi "California" và "founded" thì không.
- **Strategy A:** chỉ token đầu của mỗi word mang nhãn, các `##` subword và special token nhận IGNORE (`-100`):

$$
L = -\sum_{i:\, y_i \ne \text{IGNORE}} \log P(y_i \mid x)
$$

- Loss tính trên đúng 7 vị trí, đúng bằng số word. Mỗi word đóng góp một lần dù dài bao nhiêu subword, nên các tên dài không bị tính trọng số lớn hơn.
- Khi dự đoán, lấy nhãn ở token đầu của mỗi word, nên kết quả quay về đúng mức word và so được với gold. Các token `##on` vẫn tham gia attention, chỉ không có nhãn riêng.

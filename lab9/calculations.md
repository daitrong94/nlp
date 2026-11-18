# Calculations

## Bài 1 - BIO

| Elon | Musk | founded | SpaceX | in | California | . |
|---|---|---|---|---|---|---|
| B-PER | I-PER | O | B-ORG | O | B-LOC | O |

Từ đầu entity nhận `B-`, các từ sau trong cùng entity nhận `I-`, từ ngoài entity nhận `O`.

## Bài 2 - Entity boundary

| University | of | California | Berkeley |
|---|---|---|---|
| B-ORG | I-ORG | I-ORG | I-ORG |

**Vì sao không gán tất cả là ORG?** Vì khi đó không biết đây là một entity 4 từ hay hai entity đứng cạnh nhau. Ví dụ "gave [Microsoft] [Google] shares": BIO ghi `B-ORG B-ORG`, còn khi chỉ dùng nhãn loại thì hai entity dính thành một. NER phải nhận ra cả loại lẫn ranh giới.

## Bài 3 - Shape

$B = 8$, $T = 128$, $d = 768$, $C = 9$:

- Hidden states $H$: $8 \times 128 \times 768$
- Logits $Z = HW + b$: $8 \times 128 \times 9$ (head $W$ có kích thước $768 \times 9$, dùng chung cho mọi token)
- Labels $Y$: $8 \times 128$ (`-100` ở vị trí không tính loss)

LAB 08 chỉ lấy vector [CLS] nên logits là $8 \times C$. NER áp head lên **mọi** token.

## Label alignment (Strategy A)

| Token | El | ##on | Mu | ##sk | founded | Space | ##X | in | California | . |
|---|---|---|---|---|---|---|---|---|---|---|
| Label | B-PER | IGN | I-PER | IGN | O | B-ORG | IGN | O | B-LOC | O |

Chỉ token đầu của mỗi word mang nhãn, các subword `##` nhận IGNORE (`-100`), nên không tính vào loss:

$$
L = -\sum_{i:\, y_i \ne \text{IGNORE}} \log P(y_i \mid x)
$$

Số vị trí tính loss đúng bằng số word (7). Khi dự đoán, lấy nhãn ở token đầu là quay về được mức word.

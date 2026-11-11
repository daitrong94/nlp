# Calculations

## Bài 1 - MLM loss

Câu "The cat [MASK] fish.", token đúng là **eats**, model dự đoán $P(\text{eats}) = 0.70$:

$$
L = -\log P(\text{eats}) = -\ln 0.70 = \boxed{0.357}
$$

| $P(\text{eats})$ | Loss | Nhận xét |
|---|---|---|
| 0.95 | $-\ln 0.95 = 0.051$ | Đoán đúng và tự tin: loss gần 0 |
| 0.70 | 0.357 | |
| 0.01 | $-\ln 0.01 = 4.605$ | Gần như bỏ qua token đúng: loss rất lớn |

**Vì sao dùng cross-entropy cho MLM?** MLM là bài toán chọn một token trong vocabulary (30,522 token), tức classification nhiều lớp. Cross-entropy chỉ phụ thuộc xác suất của token đúng, phạt rất nặng khi model tự tin sai ($-\log p \to \infty$ khi $p \to 0$), và có gradient đơn giản ($p - y$) nên dễ tối ưu.

Kiểm tra với BERT-mini thật: với câu "The cat [MASK] fish.", model xếp "eats" đứng đầu với $P = 0.098$, nên loss là $-\ln 0.098 = 2.32$. Model nhỏ chưa tự tin, nhưng đã chọn đúng.

## Bài 2 - Tokenization: "The cats are playing."

Tokenizer `bert-base-uncased` (WordPiece, dùng chung vocabulary với BERT-mini):

| | |
|---|---|
| Token sequence | `[CLS] the cats are playing . [SEP]` |
| Token IDs | `[101, 1996, 8870, 2024, 2652, 1012, 102]` |
| Special tokens | `[CLS]` (101) ở đầu, `[SEP]` (102) ở cuối |
| Attention mask | `[1, 1, 1, 1, 1, 1, 1]`, không có padding |

- Words: 4 (The, cats, are, playing).
- Tokens: 7, gồm 5 token nội dung (4 từ và dấu ".") và 2 special token.
- Subword tokens: 0. Cả 4 từ đều có sẵn trong vocabulary nên không bị tách. Tokenizer cũng lowercase toàn bộ ("The" → "the").

Ví dụ có tách subword: "unwatchable" → `un ##watch ##able`, "tokenization" → `token ##ization`. Dấu `##` nghĩa là "nối tiếp token trước". Nhờ vậy từ hiếm không bị biến thành `[UNK]` như ở vocabulary theo từ của LAB 05 - 07.

Khi pad một batch, ví dụ `[CLS] the cat eats fish [SEP] [PAD] [PAD]`, attention mask là `[1, 1, 1, 1, 1, 1, 0, 0]`, nên các vị trí `[PAD]` không được attend tới.

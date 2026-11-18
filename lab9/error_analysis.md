# Error analysis

Model: DistilBERT fine-tune, test CoNLL-2003 (5,648 entity).

| Model | Đúng | Sai loại | Sai ranh giới | Thừa (FP) | Thiếu (FN) |
|---|---|---|---|---|---|
| Baseline | 4,491 | 520 | 711 | 241 | 222 |
| Transformer | 5,101 | 253 | 209 | 199 | 105 |

Transformer hiếm khi bỏ sót entity. Lỗi chủ yếu là sai loại và sai ranh giới.

## False positives

| Text | Gold | Pred | Error type | Explanation |
|---|---|---|---|---|
| State Council 's **Port Office** | O | ORG | spurious | Cụm viết hoa giữa danh sách cơ quan, gold có thể thiếu nhãn |
| to visit **Pope** | O | PER | context | "Pope" là chức danh, CoNLL không gán nhãn |
| The **East** / West spread | O | LOC | ambiguous | Thuật ngữ thị trường, không phải địa danh |
| **LATE** GOALS GIVE JAPAN WIN | O | PER | subword | Headline in hoa, `LA ##TE` không có nghĩa |
| last year 's **AFC** finalists | O | ORG | rare entity | Từ viết tắt chưa gặp trong train |

## False negatives

| Text | Gold | Pred | Error type | Explanation |
|---|---|---|---|---|
| **CHICAGO** , Dec 6 | LOC | O | subword | Gặp 59 lần trong train nhưng dạng in hoa bị tách vụn |
| **Association** games on Friday | ORG | O | context | Danh từ chung dùng làm tên riêng |
| **Network** operators said | MISC | O | ambiguous | Đứng đầu câu nên chữ hoa không còn là tín hiệu |
| BETTING ON **REAL MADRID** | ORG | O | subword | Headline in hoa |
| Iowa-S **Minn** feedlot cattle | LOC | O | rare / domain | Viết tắt của Minnesota trong bản tin thị trường gia súc |

## Boundary errors

| Text | Gold | Pred | Explanation |
|---|---|---|---|
| **Ohio State** left tackle | [Ohio State]/ORG | [Ohio]/LOC [State]/ORG | "Ohio" quen là LOC nên bị tách ra |
| **NYMEX Alberta** natgas | [NYMEX]/ORG [Alberta]/LOC | [NYMEX Alberta]/ORG | Gộp nhầm hai entity đứng liền nhau |
| **Zywiec Full Light** | [Zywiec]/ORG [Full Light]/MISC | [Zywiec Full Light]/ORG | Tên hãng và tên sản phẩm, đều chưa gặp |
| third **World cup** event | [World cup]/MISC | [World]/MISC | "cup" viết thường nên entity bị dừng sớm |
| Syria - **Nader Jokhadar** 7 | [Nader Jokhadar]/PER | [Nader]/ORG [Jokhadar]/PER | Tên hiếm, không có ngữ cảnh câu |

## Nguyên nhân chính

- **Tokenization:** recall trên headline in hoa chỉ 0.81, so với 0.92 trên câu thường.
- **Entity hiếm:** recall 0.86 với entity chưa gặp, 0.96 với entity gặp từ 10 lần.
- **Loại mơ hồ:** LOC ↔ ORG nhầm 85 lần, ví dụ tên thành phố cũng là tên đội bóng.
- **Gold chưa nhất quán:** một số "lỗi" có thể do annotation ("Port Office", "AFC").

# Error analysis

Model: DistilBERT-base-cased fine-tune trên CoNLL-2003, đánh giá trên test (3,453 câu, 5,648 gold entity). Các ví dụ được lấy ngẫu nhiên (seed cố định) trong `ner_finetuning.ipynb`.

## Tổng quan lỗi (mức entity)

| Model | Đúng | Wrong type | Boundary | Spurious (FP) | Missing (FN) |
|---|---|---|---|---|---|
| Baseline | 4,491 | 520 | 711 | 241 | 222 |
| Transformer | 5,101 | 253 | 209 | 199 | 105 |

- Transformer hiếm khi bỏ sót entity. Lỗi chủ yếu là **sai loại** và **sai ranh giới**.
- Baseline sai ranh giới nhiều gấp 3.4 lần, vì nó phân loại từng word độc lập.

## False positives (spurious entity)

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| "the State Council 's **Port Office** , the Civil Aviation Administration" | O | ORG (0.94) | spurious, label ambiguity | Cụm viết hoa nằm giữa danh sách cơ quan nên trông giống hệt một ORG. Gold không gán, có thể do annotation thiếu nhất quán. Baseline cũng đoán ORG. |
| "ex-communist president to visit **Pope**" | O | PER (0.97) | spurious, context-dependent | "Pope" là chức danh, CoNLL không gán. Model thấy từ viết hoa chỉ một người sau "visit" nên đoán PER. |
| "The **East** / West spread narrowed" | O | LOC (0.98) | spurious, ambiguous entity | "East" thường là một phần của địa danh. Ở đây nó là thuật ngữ thị trường (domain tài chính), model không có tín hiệu để phân biệt. |
| "SOCCER - **LATE** GOALS GIVE JAPAN WIN" | O | PER (0.39) | spurious, subword alignment | Headline in hoa: "LATE" bị tách thành `LA ##TE`, token đầu "LA" vô nghĩa. Confidence thấp cho thấy model đang đoán. |
| "last year 's **AFC** finalists" | O | ORG (0.74) | spurious, rare entity | Từ viết tắt in hoa chưa gặp trong train. Hình thức giống ORG, nhưng gold không gán (ở đây là tên giải đấu). |

## False negatives (missing entity)

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| "**CHICAGO** , Dec 6 ( Reuter )" | LOC | O (0.62) | missing, subword alignment | Dateline in hoa bị tách thành nhiều mảnh. Từ này gặp 59 lần trong train và baseline đoán đúng, nên lỗi do tokenization chứ không do hiếm. |
| "**Association** games on Friday" | ORG | O (1.00) | missing, context-dependent | Danh từ chung dùng làm tên riêng (viết tắt của tên giải). Model rất tự tin là O. |
| "**Network** operators said the draft laws" | MISC | O (0.99) | missing, ambiguous entity | Từ thường đứng đầu câu nên chữ hoa không còn là tín hiệu. Nhãn MISC ở đây cũng khó đoán. |
| "BETTING ON **REAL MADRID** V BARCELONA" | ORG | O (0.60) | missing, subword alignment | Headline in hoa. "REAL" in hoa trông như từ thường, model bỏ sót cả cụm. |
| "Iowa-S **Minn** feedlot cattle market" | LOC | O (0.78) | missing, rare entity, domain | Viết tắt của Minnesota trong bản tin thị trường gia súc. Chỉ gặp 1 lần trong train, văn phong khác tin tức thường. |

## Boundary errors

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| "**Ohio State** left tackle Orlando Pace" | [Ohio State]/ORG | [Ohio]/LOC + [State]/ORG | wrong boundary | "Ohio" gặp 15 lần trong train với nhãn LOC, nên model tách tên trường thành địa danh + tổ chức. |
| "**NYMEX Alberta** natgas remained untraded" | [NYMEX]/ORG + [Alberta]/LOC | [NYMEX Alberta]/ORG | wrong boundary (gộp) | Hai entity khác loại đứng liền nhau, không có dấu phân cách. Baseline đoán đúng nhờ feature riêng từng từ. |
| "leading brand , **Zywiec Full Light** ," | [Zywiec]/ORG + [Full Light]/MISC | [Zywiec Full Light]/ORG | wrong boundary (gộp), rare entity | Tên hãng + tên sản phẩm, cả hai chưa gặp. Model coi cả cụm viết hoa là một tên. |
| "the third **World cup** event" | [World cup]/MISC | [World]/MISC | wrong boundary (cắt thiếu) | "cup" viết thường nên model dừng entity sớm. Tín hiệu viết hoa lấn át ngữ nghĩa. |
| "Syria - **Nader Jokhadar** 7" | [Nader Jokhadar]/PER | [Nader]/ORG + [Jokhadar]/PER | wrong boundary + wrong type | Dòng liệt kê cầu thủ ghi bàn, không có ngữ cảnh câu. Tên hiếm bị tách nhiều subword, "Nader" đứng sau tên đội nên bị đoán ORG. |

## Wrong entity type

| Text | Gold | Prediction | Explanation |
|---|---|---|---|
| "a ( **Santa Fe** ) deal with someone" | ORG | LOC (0.96) | Tên công ty trùng địa danh, câu ngắn không đủ ngữ cảnh. |
| "**Barrick** did not hold talks" | ORG | PER (0.91) | Tên công ty giống họ người, "hold talks" hợp với cả người lẫn tổ chức. |
| "the **London-to-Boston** route" | MISC | LOC (0.99) | Từ ghép từ hai địa danh. Nhãn MISC là quy ước annotation khó đoán. |

## Tổng hợp nguyên nhân

- **Tokenization:** recall trên câu in hoa toàn bộ là 0.806, so với 0.915 trên câu thường. "CHICAGO" bị bỏ sót dù gặp 59 lần trong train.
- **Entity hiếm:** recall 0.863 với entity chưa gặp, 0.965 với entity gặp từ 10 lần.
- **Loại mơ hồ:** LOC ↔ ORG nhầm 85 lần, MISC ↔ ORG 60 lần. MISC có F1 thấp nhất (0.763).
- **Ranh giới:** model dựa mạnh vào chữ hoa để quyết định entity kết thúc ở đâu.
- **Label ambiguity:** một số "lỗi" đáng ngờ là do gold ("Port Office", "AFC"), nên F1 0.894 chưa chắc phản ánh hết năng lực model.
- **Confidence:** nhiều lỗi có confidence trên 0.9, nên không thể chỉ dùng confidence để lọc lỗi.

# Prediction

Ghi trước khi chạy pretrained model. Kết quả đối chiếu ở mục cuối.

## Prediction 1 - Pretrained + fine-tuning có tốt hơn Transformer train from scratch không?

- Prediction: Có. Cùng kiến trúc (BERT-mini, 4 layer), pretrained sẽ hơn from scratch khoảng 5 - 10 điểm F1.
- Reason: Model pretrained đã học biểu diễn ngôn ngữ (nghĩa từ, ngữ cảnh) từ lượng văn bản rất lớn. Model from scratch phải tự học mọi thứ từ 20K review. Ở LAB 07, Transformer from scratch chỉ đạt F1 khoảng 0.81.
- Expected evidence: F1 trên test, và pretrained hội tụ nhanh hơn (validation loss thấp ngay từ epoch đầu).

## Prediction 2 - Freeze encoder, chỉ train classification head

- Prediction: **C. Kém hơn** fine-tuning toàn bộ, nhưng vẫn tốt hơn mức đoán ngẫu nhiên nhiều (khoảng 0.70 - 0.80 F1).
- Reason: Biểu diễn [CLS] của model pretrained chưa được học cho sentiment (MLM không dạy phân biệt positive/negative), nên một lớp Linear trên đó chỉ khai thác được một phần thông tin. Fine-tuning cho phép encoder điều chỉnh biểu diễn theo task.

## Prediction 3 - Chỉ có 10% training data, model nào giảm mạnh hơn?

- Prediction: Model from scratch giảm mạnh hơn.
- Reason: Pretrained model đã có sẵn biểu diễn ngôn ngữ, nên dữ liệu downstream chỉ cần để học ranh giới phân loại. Model from scratch phải học cả biểu diễn từ (embedding) lẫn task từ dữ liệu ít ỏi, nên với 2,000 review sẽ thiếu dữ liệu nghiêm trọng.

## Đối chiếu sau experiment

Phần prediction ở trên giữ nguyên như đã ghi trước khi chạy.

| | Prediction | Kết quả | Đúng? |
|---|---|---|---|
| P1 | Pretrained hơn from scratch 5 - 10 điểm F1 | 0.843 so với 0.814, hơn 2.9 điểm | Đúng chiều, sai độ lớn |
| P2 | C: frozen kém hơn, F1 khoảng 0.70 - 0.80 | Frozen 0.736 (mean pooling 0.745), fine-tuning 0.843 | Đúng |
| P3 | From scratch giảm mạnh hơn khi chỉ có 10% data | Pretrained giảm 5.3 điểm (0.841 → 0.788), from scratch giảm 67.7 điểm (0.808 → 0.131) | Đúng |

**Vì sao P1 sai về độ lớn?**

- Với đủ 20K review, model from scratch vẫn học được các từ khoá sentiment, giống LAB 07 (F1 khoảng 0.81). Ưu thế của pretraining chủ yếu thể hiện khi data ít: ở 25% data, khoảng cách là 9.5 điểm.
- BERT-mini là model nhỏ (4 layer, hidden 256). Max_length 128 lại cắt mất phần lớn review, nên phần trần mà pretraining có thể kéo lên bị giới hạn: khi tăng lên 256 token, F1 tăng thêm khoảng 5 điểm.
- Hai model chỉ được train 2 epoch. Trong đó model pretrained hội tụ nhanh hơn rõ rệt: sau epoch 1, val_acc là 0.838 so với 0.820.

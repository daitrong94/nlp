# Prediction

Ghi trước khi chạy model.

## P1 - Transformer so với baseline dùng word features

- Prediction: Transformer hơn khoảng 10 - 15 điểm F1 (baseline 0.75 - 0.80, Transformer 0.88 - 0.91).
- Reason: baseline chỉ thấy chính từ đó và ±2 từ, nên phải nhớ tên. Transformer có ngữ cảnh cả câu, nên nhận ra được tên mới.

## P2 - Entity dài có khó hơn entity một từ?

- Prediction: Khó hơn, F1 giảm dần từ 1 đến 4+ từ.
- Reason: entity chỉ đúng khi mọi từ đều đúng. Nếu mỗi từ đúng với xác suất $p$ thì cả entity $n$ từ đúng với xác suất khoảng $p^n$.

## P3 - Entity hiếm có gây lỗi nhiều hơn?

- Prediction: Có, và baseline bị ảnh hưởng nặng hơn.
- Reason: entity đã gặp thì model nhớ được. Entity chưa gặp thì model chỉ còn dựa vào chữ hoa và ngữ cảnh.

## Đối chiếu sau experiment

| | Kết quả | Đúng? |
|---|---|---|
| P1 | 0.774 so với 0.894 (+12.1 điểm) | Đúng |
| P2 | Baseline: 0.76 / 0.84 / 0.65 / 0.47. Transformer: 0.89 / 0.91 / 0.85 / 0.86 | Đúng với baseline, sai với Transformer |
| P3 | Recall với entity chưa gặp: Transformer 0.86, baseline 0.66. Với entity gặp từ 10 lần: cả hai 0.96 | Đúng |

**Vì sao P2 sai với Transformer?** Lập luận $p^n$ giả định các từ được đoán độc lập, điều này chỉ đúng với baseline. Transformer dùng self-attention nên thấy cả cụm, và các nhãn trong một entity nhất quán với nhau.

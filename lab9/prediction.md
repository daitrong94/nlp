# Prediction

Ghi trước khi chạy baseline và Transformer. Kết quả đối chiếu ở mục cuối.

## Prediction 1 - Pretrained Transformer so với classifier dùng word features

- Prediction: Transformer tốt hơn rõ, entity-level F1 cao hơn baseline khoảng 10 - 15 điểm (baseline khoảng 0.75 - 0.80, Transformer khoảng 0.88 - 0.91). Phần lớn khoảng cách nằm ở **recall** và ở các entity chưa gặp khi train.
- Reason: baseline Logistic Regression chỉ nhìn chính từ đó, dạng chữ (viết hoa, có số) và một từ hai bên, mỗi từ được phân loại độc lập. Nó gần như phải nhớ tên, nên khi gặp tên mới chỉ còn dựa vào chữ hoa. Transformer pretrained có biểu diễn contextual từ cả câu, nên phân biệt được "Washington" là người hay địa điểm, và nhận ra tên mới qua ngữ cảnh (ví dụ "said", "in", "at").
- Expected evidence: F1 theo loại entity và theo tần suất xuất hiện trong train. Khoảng cách lớn nhất nên ở ORG, MISC và entity hiếm.

## Prediction 2 - Entity dài nhiều từ dễ hay khó hơn entity một từ?

- Prediction: **Khó hơn.** F1 giảm dần theo độ dài: 1 từ > 2 từ > 3 từ > 4+ từ.
- Reason: entity chỉ được tính đúng khi **mọi** từ đều đúng nhãn, kể cả ranh giới. Nếu mỗi từ đúng với xác suất $p$ thì cả entity $n$ từ đúng với xác suất khoảng $p^n$. Entity dài như "United Nations Security Council" còn chứa từ thường ("of", "for") và dễ bị cắt ranh giới sai, ví dụ chỉ nhận "Security Council". Riêng entity 2 từ thường là tên người (họ + tên) rất đều đặn, nên có thể không khó hơn 1 từ nhiều.

## Prediction 3 - Entity hiếm trong training data có gây lỗi nhiều hơn không?

- Prediction: **Có.** Entity không xuất hiện trong train sẽ có recall thấp hơn rõ so với entity xuất hiện nhiều lần, và baseline bị ảnh hưởng nặng hơn Transformer.
- Reason: với entity đã gặp, model có thể nhớ cả tên lẫn loại. Với entity chưa gặp, model phải suy ra từ hình thức (viết hoa, hậu tố như "-ville", "Inc") và ngữ cảnh. Loại entity của tên mới còn mơ hồ: một tên viết hoa có thể là PER, ORG hay LOC. Transformer đỡ hơn vì đã thấy nhiều tên trong pretraining và dùng được ngữ cảnh, nhưng vẫn sai nhiều hơn ở entity hiếm.

## Đối chiếu sau experiment

Phần prediction ở trên giữ nguyên như đã ghi trước khi chạy.

| | Prediction | Kết quả | Đúng? |
|---|---|---|---|
| P1 | Transformer hơn baseline 10 - 15 điểm F1 (0.75 - 0.80 so với 0.88 - 0.91) | 0.774 so với 0.894, hơn 12.1 điểm | Đúng |
| P2 | Entity càng dài càng khó: 1 > 2 > 3 > 4+ word | Baseline: 0.756 / 0.836 / 0.649 / 0.468. Transformer: 0.890 / 0.908 / 0.854 / 0.862 | Đúng với baseline, gần như sai với Transformer |
| P3 | Entity hiếm sai nhiều hơn, baseline bị nặng hơn | Recall chưa gặp so với gặp ≥ 10 lần: Transformer 0.863 so với 0.965, baseline 0.664 so với 0.962 | Đúng |

**P1.** Đúng cả về chi tiết: khoảng cách lớn nhất nằm ở ORG (+19.4 điểm) và ở entity chưa gặp trong train (+19.9 điểm recall). Riêng MISC thì không như dự đoán, gần như không cải thiện (0.761 → 0.763).

**Vì sao P2 sai với Transformer?**

- Lập luận $p^n$ giả định các word được đoán độc lập. Điều đó đúng với baseline, nên baseline giảm mạnh (0.468 ở 4+ word). Transformer thì không đoán độc lập: self-attention cho mỗi token thấy cả cụm, nên các nhãn trong một entity nhất quán với nhau.
- Entity dài thường có cấu trúc dễ nhận, ví dụ "... of ...", "... Corp". Trong khi đó entity 1 word lại hay mơ hồ về loại ("Barcelona" là LOC hay ORG), nên nhóm 1 word (0.890) không phải nhóm dễ nhất.
- Phần đúng của dự đoán: entity 2 word (đa số là tên người) là nhóm cao nhất ở cả hai model. Nhóm 4+ chỉ có 62 entity nên chưa đủ để kết luận chắc.

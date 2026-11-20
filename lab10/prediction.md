# Prediction

Ghi trước khi chạy model.

## P1 - Question và context trùng nhiều từ, nhưng chỉ một chỗ là đáp án

- Prediction: Có, model dễ bị nhầm sang câu trùng từ nhưng không chứa đáp án. Baseline bị nhầm nặng hơn nhiều.
- Reason: SQuAD có nhiều câu hỏi được viết lại gần như nguyên văn từ context, nên model học được mẹo "tìm chỗ trùng từ". Khi câu hỏi diễn đạt khác context, mẹo này mất tác dụng.

## P2 - Answer càng dài thì boundary càng khó?

- Prediction: Có. EM giảm mạnh theo độ dài, F1 giảm ít hơn.
- Reason: answer dài có nhiều điểm hợp lý để bắt đầu và kết thúc (có hay không có mạo từ, mệnh đề phụ). Lệch một token là EM = 0, nhưng F1 vẫn được điểm phần trùng.

## P3 - Context chứa thông tin không liên quan và câu gây nhiễu

- Prediction: F1 giảm khoảng 5 - 15 điểm khi chèn một câu gây nhiễu (giống câu hỏi nhưng sai đáp án). Thông tin không liên quan hẳn thì gần như không ảnh hưởng.
- Reason: câu gây nhiễu có nhiều từ trùng với câu hỏi nên kéo model về phía nó. Câu không liên quan thì không có tín hiệu nào hấp dẫn model.

## P4 - Đổi thứ tự các câu trong context, giữ nguyên answer

- Prediction: Ảnh hưởng nhỏ, F1 giảm khoảng 1 - 3 điểm.
- Reason: đáp án thường nằm trong một câu, và self-attention nhìn được mọi vị trí. Chỉ những câu hỏi cần nối thông tin giữa các câu, hoặc có đại từ trỏ về câu trước, mới bị ảnh hưởng.

## Đối chiếu sau experiment

| | Dự đoán | Kết quả | Đúng? |
|---|---|---|---|
| P1 | Overlap cao dễ hơn, baseline phụ thuộc hơn | Transformer F1 75.8 → 88.9; baseline 12.5 → 32.4 | Đúng |
| P2 | Answer dài khó hơn, EM giảm mạnh hơn F1 | EM −27.6, F1 −13.6 (1 từ → 7+ từ) | Đúng |
| P3 | Distractor −5 đến −15 F1; irrelevant ≈ 0 | Distractor **−62.9** F1; irrelevant −0.5 | Đúng hướng, sai mức độ |
| P4 | Reorder −1 đến −3 F1 | −1.9 F1 | Đúng |

Điều bất ngờ nhất là P3: model dễ bị câu gây nhiễu đánh lừa hơn mình nghĩ rất nhiều. 75% đáp án rơi vào câu nhiễu.

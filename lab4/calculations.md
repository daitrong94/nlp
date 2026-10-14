# Calculations

## Bài 1 - Binary classification

x = [1, 2, 0], w = [0.5, -0.2, 0.4], b = 0.1

z = wᵀx + b = 0.5·1 + (-0.2)·2 + 0.4·0 + 0.1 = 0.5 - 0.4 + 0 + 0.1 = 0.2

P(y = 1|x) = σ(0.2) = 1/(1 + e^(-0.2)) = 1/(1 + 0.8187) ≈ 0.550

## Bài 2 - Prediction threshold

P(y = 1|x) = 0.72

- τ = 0.5: 0.72 ≥ 0.5, model dự đoán class 1 (positive).
- τ = 0.8: 0.72 < 0.8, model dự đoán class 0 (negative).

Tăng threshold thì model dự đoán positive ít hơn và chỉ giữ các trường hợp chắc chắn. FP giảm nên precision thường tăng, FN tăng nên recall giảm. Giảm threshold thì ngược lại: recall tăng, precision giảm.

## Bài 3 - Confusion matrix

TP = 80, FN = 20, FP = 10, TN = 90

- Precision = TP/(TP + FP) = 80/90 ≈ 0.889
- Recall = TP/(TP + FN) = 80/100 = 0.800
- F1 = 2·P·R/(P + R) = 2·0.889·0.8/(0.889 + 0.8) ≈ 0.842
- (Accuracy = (80 + 90)/200 = 0.850)

Với bài toán phát hiện spam (positive = spam):

- False positive: email bình thường bị đưa vào thư mục spam. Người dùng có thể bỏ lỡ email quan trọng, đây thường là lỗi nặng hơn.
- False negative: email spam lọt vào inbox. Gây phiền nhưng ít hậu quả hơn.

Vì vậy bộ lọc spam thường ưu tiên precision cao (FP thấp).

## Bài 4 - Accuracy trap

950 negative, 50 positive, model dự đoán mọi sample là negative: TP = 0, FN = 50, FP = 0, TN = 950.

- Accuracy = 950/1000 = 95%
- Recall (positive) = 0/50 = 0
- Precision (positive) = 0/0, không xác định (quy ước 0), model không dự đoán positive lần nào
- F1 (positive) = 0
- F1 (negative) = 2·950/(2·950 + 50) ≈ 0.974, macro F1 = (0 + 0.974)/2 ≈ 0.487
- Balanced accuracy = (0 + 1)/2 = 0.5

Model này không tốt: nó không phát hiện được positive nào, và 95% chỉ bằng đúng baseline "đoán class đa số". Với dữ liệu mất cân bằng cần xem thêm confusion matrix, precision, recall và F1 của class thiểu số (hoặc macro F1, balanced accuracy).

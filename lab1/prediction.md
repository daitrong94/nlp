# Predictions — Part C

> Ghi lại trước khi mở corpus 30K documents. Kết quả thực nghiệm chỉ được xem sau khi đã ghi prediction.

## Prediction 1 — Vocabulary

Với 30K documents là văn bản web đa chủ đề, vocabulary có thể lên tới vài chục nghìn đến hơn một trăm nghìn unique terms - vì web text có nhiều tên riêng, số, lỗi chính tả, biến thể từ, không như văn bản chuyên ngành hẹp có vocab giới hạn hơn.

## Prediction 2 — Sparsity

TF-IDF matrix sẽ rất **sparse**. Mỗi document chỉ dùng vài trăm từ trong khi vocab toàn corpus lớn hơn rất nhiều, nên tỷ lệ zero entries có thể trên **99%**.

## Prediction 3 — Search

Không nhất thiết. TF-IDF/cosine similarity chỉ đo overlap từ vựng (lexical), nên document đứng đầu là document **dùng nhiều từ giống query nhất**, chứ không hẳn là document **gần nghĩa nhất** - hai document có thể diễn đạt cùng ý nhưng dùng từ khác nhau (đồng nghĩa) sẽ bị xếp thấp, ngược lại document dùng đúng từ nhưng khác ngữ cảnh có thể bị xếp cao nhầm.

## Exercise 6 — Prediction (medical image classification)

1. Document similarity cao nhất: D1 (trùng hoàn toàn với Q).
2. Document similarity thấp nhất: D3 (không chung từ nào với Q).
3. Term có thể IDF thấp: "medical", "image" (xuất hiện ở cả D1 và D2).
4. Nếu bỏ IDF, ranking tổng thể không đổi nhiều (D1 vẫn đứng đầu), nhưng khoảng cách D1–D2 sẽ thu hẹp vì mất trọng số phân biệt của từ đặc trưng "classification".

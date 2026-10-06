# Prediction

Prediction, Reason và Confidence được ghi trước khi chạy notebook. Dòng "Sau experiment" ghi sau khi có kết quả.

## Prediction 1 - Từ nào gần nhau nhất? (doctor, physician, hospital, banana, car)

- Prediction: doctor–physician gần nhất, sau đó là hospital. Banana và car xa nhóm y tế.
- Reason: doctor và physician thay thế được cho nhau trong câu. Hospital chỉ liên quan về chủ đề.
- Confidence: 85%
- Sau experiment: Đúng phần lớn. doctor–physician = 0.73, physician–hospital = 0.60, doctor–hospital = 0.56, banana ≈ 0. Sai ở car: doctor–car = 0.37, do cả hai hay xuất hiện trong văn bản quảng cáo/dịch vụ.

## Prediction 2 - Window 2 → 5, similarity có đổi không?

- Prediction: Có. Các cặp cùng chủ đề tăng, các cặp thay thế được giảm.
- Reason: Window lớn thấy được chủ đề, window nhỏ thấy vai trò ngữ pháp.
- Confidence: 65%
- Sau experiment: Đúng hướng, nhưng mức thay đổi nhỏ. doctor–physician 0.758 → 0.728, doctor–disease 0.371 → 0.456. WordSim353 tăng từ 0.566 lên 0.618.

## Prediction 3 - Dimension 50 → 100 → 300, chất lượng có chắc tăng không?

- Prediction: Không chắc. Từ 100 lên 300 có thể không tăng.
- Reason: Corpus khoảng 11 triệu token, không đủ để dùng hết capacity của 300 chiều.
- Confidence: 60%
- Sau experiment: Đúng. WordSim353 (0.571 / 0.618 / 0.605) và analogy (0.224 / 0.277 / 0.254) đều tốt nhất ở 100 chiều. Model size tăng 20 → 41 → 123 MB.

## Prediction 4 - Corpus 100 câu, doctor và physician có chắc gần nhau không?

- Prediction: Không, similarity sẽ không đáng tin và dao động theo seed.
- Reason: Mỗi từ chỉ có vài chục context.
- Confidence: 80%
- Sau experiment: Đúng ở kết luận chính, sai ở ý dao động theo seed. Trong 100 câu ngẫu nhiên, cả hai từ không xuất hiện lần nào. Với 100 câu chọn sẵn, cos = 0.998 và ổn định qua các seed, nhưng cosine trung bình của doctor với mọi từ cũng đã là 0.96, nên con số này vô nghĩa. Physician chỉ vào top 5 láng giềng của doctor khi có từ 10,000 document trở lên.

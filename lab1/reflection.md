# Reflection

> Tối đa ~500 từ. Không dùng AI cho phần nội dung phân tích/reflection.

1. Prediction sai: vocabulary size - dự đoán thấp hơn thực tế (193,540 từ). Sparsity dự đoán đúng (~99.9%).

2. Bất ngờ nhất: bỏ stopword (Pipeline B) làm P@5 giảm ở query "school students education" (0.80->0.60), trái với giả định "bỏ stopword luôn tốt hơn".

3. Evidence mạnh nhất: Experiment 2 (ablation) - so sánh trực tiếp nhiều pipeline trên cùng query, cho thấy preprocessing là trade-off chứ không phải "càng nhiều càng tốt".

4. Failure case quan trọng nhất: query "online shopping discount" - hệ thống xếp cao doc về cước SMS chỉ vì trùng từ "discount", cho thấy TF-IDF không phân biệt được nghĩa khác nhau của cùng một từ.

5. Nếu xây lại: thêm phrase/n-gram matching để giảm false positive do trùng từ đơn lẻ khác ngữ cảnh.

6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?

```
AI contribution:
- Code: implementation.py (core TF-IDF/cosine functions + tests) and all
  code in experiments.ipynb (data loading, pipeline, ablation, search, eval).
- Selected the 6 evaluation queries and assigned relevance labels based on
  reading the search result previews.
- Drafted the content of calculations.md, prediction.md, the analysis
  answers in experiments.ipynb, the error analysis (Part I), and this
  reflection.
- I reviewed, verified, and confirmed my understanding of the above content
  before submission.
```

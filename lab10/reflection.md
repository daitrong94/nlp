# Reflection

## LAB 08, 09, 10: cùng backbone, khác head

```
                 Pretrained Transformer (encoder)
                              ↓
               H: một vector contextual cho mỗi token
          ↓                   ↓                    ↓
   vector [CLS]          mọi token            mọi token context
          ↓                   ↓                    ↓
 Linear d → C          Linear d → số nhãn    Linear d → 2 (start, end)
          ↓                   ↓                    ↓
 Classification        NER (BIO)             Start / End span
     LAB 08              LAB 09                LAB 10
```

Ba lab dùng cùng một encoder đã pretrain. Với một chuỗi $T$ token, encoder trả về ma trận $H$ kích thước $T \times d$, trong đó mỗi vector đã chứa ngữ cảnh của cả chuỗi nhờ self-attention. Cái khác nhau là **câu hỏi mà task đặt ra cho $H$**:

- **LAB 08:** "cả review là positive hay negative?". Chỉ cần một vector đại diện cho cả chuỗi, nên lấy [CLS] rồi phân loại.
- **LAB 09:** "mỗi từ thuộc loại entity nào?". Cần một nhãn cho từng token, nên áp cùng một Linear lên mọi vector.
- **LAB 10:** "đoạn nào trả lời câu hỏi?". Output không phải nhãn mà là **vị trí**. Head cho mỗi token hai điểm số, sau đó softmax trên toàn chuỗi, được $P_{start}$ và $P_{end}$. Câu hỏi và context đi chung một input, nên attention nối được từ trong câu hỏi với từ trong context. Khi dự đoán còn một bước hậu xử lý: chọn cặp start ≤ end có tổng điểm cao nhất, trên mọi window của câu hỏi.

Head rất nhỏ: head QA chỉ có $768 \times 2 + 2 = 1,538$ tham số, trên tổng 65 triệu. Kiến thức ngôn ngữ nằm gần như hết trong backbone. Vì vậy fine-tune 15 phút trên T4 đã đạt F1 82.9, trong khi baseline trùng từ chỉ đạt 22.4.

Đổi task kéo theo đổi target và đổi cách đánh giá. LAB 09 phải căn nhãn word sang subword. LAB 10 phải đổi vị trí ký tự sang vị trí token bằng offset mapping, và chia context dài thành window. Metric cũng đổi: accuracy, rồi entity F1, rồi EM/F1 trên span.

Backbone mạnh không có nghĩa là model hiểu. Khi mình thêm một câu nhiễu chép lại câu hỏi với một đáp án sai, F1 sụp từ 86.6 xuống 23.7. Model vẫn dựa nhiều vào chỗ trùng từ với câu hỏi.

## Individual learning check

1. **Extractive QA khác classification?** Classification chọn một nhãn trong tập cố định. Extractive QA chọn một đoạn (span) nằm trong context.
2. **Vì sao cần start và end?** Một span được xác định bởi hai vị trí. Dự đoán riêng hai vị trí chỉ cần $2T$ điểm số thay vì $T^2$.
3. **Character offset khác token index?** Character offset đếm ký tự trong chuỗi gốc. Token index đếm token sau khi tokenize.
4. **Vì sao cần offset mapping?** Để đổi answer từ ký tự sang token khi train, và từ token về ký tự khi trả kết quả.
5. **stride dùng làm gì?** Cho các window chồng lên nhau, để answer nằm ở chỗ cắt vẫn trọn vẹn trong ít nhất một window.
6. **Vì sao phải chia window?** Model chỉ nhận tối đa 512 token, và attention tốn chi phí $T^2$.
7. **EM đo gì?** Prediction có trùng hoàn toàn với gold sau khi chuẩn hóa hay không.
8. **F1 của QA khác accuracy?** F1 tính độ trùng token giữa prediction và gold, nên đúng một phần vẫn được điểm.
9. **Vì sao F1 > 0 nhưng EM = 0?** Prediction lệch ranh giới, thừa hoặc thiếu vài từ ("May 1888" so với "in May 1888").
10. **Wrong start khác wrong end?** Wrong start: đầu span sai, cuối đúng. Wrong end: ngược lại.
11. **Extractive QA khác RAG?** Extractive QA cần context có sẵn và chỉ trích đoạn từ đó. RAG tự tìm tài liệu rồi sinh câu trả lời bằng lời của model.
12. **Khi nào extractive QA không trả lời được?** Khi đáp án không có trong context, khi cần tổng hợp nhiều chỗ, hoặc khi đáp án là yes/no.

## Sang LAB 11

LAB 10 giả định context chứa đáp án đã được cho trước. Thực tế ta có hàng nghìn tài liệu, nên cần một bước retrieval: tìm top-K đoạn liên quan đến câu hỏi, rồi mới đưa vào QA model.

# Reflection

## Vì sao cùng một pretrained Transformer dùng được cho cả sequence classification lẫn token classification?

Vì phần khó nhất, hiểu từng từ trong ngữ cảnh, nằm ở encoder, và encoder không gắn với task nào. Với một câu $T$ token, encoder luôn trả về ma trận $H$ kích thước $T \times d$: mỗi token có một vector contextual, được tính bằng self-attention trên cả câu. Pretraining bằng MLM buộc mọi vị trí phải mang đủ thông tin để đoán lại từ bị che, nên vector ở **mọi** vị trí đều có nghĩa, không riêng [CLS].

Hai loại task chỉ khác nhau ở cách đọc $H$:

- **Sequence classification (LAB 08):** lấy một vector, $H[0]$ của [CLS], rồi áp Linear $d \to C$. Kết quả là một nhãn cho cả câu.
- **Token classification (LAB 09):** áp cùng một Linear $d \to C$ lên **từng hàng** của $H$, $Z = HW + b$ với kích thước $T \times C$. Kết quả là một nhãn BIO cho mỗi token.

Head trong lab này chỉ có $768 \times 9 + 9 = 6921$ tham số, trên tổng 65.2M. Gần như toàn bộ kiến thức nằm trong encoder dùng chung. Ngoài head, thứ thay đổi là **cách định nghĩa label và loss**: NER cần nhãn cho từng token, nên phát sinh bài toán alignment giữa word và subword. Lab giải quyết bằng cách chỉ gán nhãn cho token đầu của mỗi word, các subword sau nhận IGNORE và không tính vào loss.

Kết quả thực nghiệm ủng hộ cách nhìn này. Sau 3 epoch fine-tune, DistilBERT đạt entity F1 0.894, so với 0.774 của baseline dùng feature thủ công. Lợi thế nằm đúng ở chỗ cần biểu diễn contextual: entity chưa gặp trong train (recall 0.863 so với 0.664), và những chuỗi mang nhiều loại như "Santa Fe" (đúng 67% so với 6%). Baseline giống biểu diễn tĩnh ở LAB 03: mỗi từ có một bộ trọng số cố định, nên phải chọn một loại cho mọi ngữ cảnh.

Tuy vậy, đổi head không giải quyết được mọi thứ. Tokenizer vẫn là giới hạn chung: trên headline in hoa toàn bộ, "GERMANY" bị tách thành 5 mảnh và recall giảm từ 0.915 xuống 0.806. Cách đánh giá cũng phải đổi theo task: token accuracy 97.9% nghe rất cao, nhưng 83% token là O, nên phải đo ở mức entity. Model còn bị domain shift: câu tin tức viết thường toàn bộ làm nó bỏ sót cả 3 entity.

Bài học là nên nhìn pretrained Transformer như một **backbone** tạo contextual representation. Task được quyết định bởi prediction head và cách định nghĩa output: một vector cho sequence task, mọi vector cho token task, và ở LAB 10 là hai phân phối start/end cho span task.

## Individual learning check

1. **Sequence classification khác token classification thế nào?** Sequence: cả câu nhận một nhãn, lấy từ vector [CLS]. Token: mỗi token nhận một nhãn, head áp lên mọi vị trí.
2. **BIO tagging dùng để làm gì?** Biến bài toán tìm span thành gán nhãn từng token, mã hóa cả loại lẫn ranh giới của entity.
3. **B-PER khác I-PER thế nào?** B-PER là từ mở đầu một tên người, I-PER là từ tiếp theo trong cùng tên đó. Nhờ vậy tách được hai tên đứng liền nhau.
4. **Tại sao tokenizer gây ra label alignment problem?** Nhãn gán theo word, nhưng tokenizer tách word thành nhiều subword ("Musk" → `Mu ##sk`), nên số token không khớp số nhãn.
5. **IGNORE trong loss nghĩa là gì?** Vị trí có nhãn -100 không được tính vào loss. Token đó vẫn tồn tại và vẫn tham gia attention.
6. **Tại sao không nên chỉ đánh giá NER bằng token accuracy?** Đa số token là O (83%), nên đoán toàn O cũng được 83%. Baseline đạt 96.2% token accuracy nhưng entity F1 chỉ 0.774.
7. **Entity-level F1 khác token-level F1 thế nào?** Entity-level chỉ tính đúng khi đúng cả loại lẫn toàn bộ ranh giới. "New York" đoán `B-LOC O` thì đúng 1 token nhưng entity sai.
8. **Vì sao entity boundary khó?** Tên dài chứa từ thường ("of", "cup"), hai entity có thể đứng liền nhau ("NYMEX Alberta"), và tên này có thể nằm trong tên khác ("Ohio State").
9. **Vì sao contextual representation hữu ích cho NER?** Cùng một chuỗi có thể là loại khác nhau tùy câu ("Barcelona" là thành phố hoặc đội bóng). Vector contextual thay đổi theo câu, vector tĩnh thì không.
10. **NER dùng thế nào trong Information Extraction?** NER tìm và phân loại entity trong văn bản thô. Gom các span lại sẽ được dữ liệu có cấu trúc, ví dụ `{"PERSON": [...], "LOCATION": [...]}`, làm đầu vào cho các bước sau như tìm quan hệ hay xây knowledge base.

## Sang LAB 10

NER trả lời "trong câu có những entity nào". Question Answering hỏi cụ thể hơn: với một câu hỏi và một đoạn văn, đáp án nằm ở **đoạn nào** trong văn bản. Backbone vẫn giữ nguyên, chỉ đổi head: thay vì 9 logit BIO cho mỗi token, model dự đoán hai phân phối $P_{start}$ và $P_{end}$ để chọn span trả lời.

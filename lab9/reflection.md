# Reflection

## Vì sao cùng một Transformer dùng được cho cả sequence classification lẫn token classification?

Vì phần khó nhất, hiểu từng từ trong ngữ cảnh, nằm ở encoder, và encoder không phụ thuộc vào task. Với một câu $T$ token, encoder luôn trả về ma trận $H$ kích thước $T \times d$: mỗi token có một vector contextual được tính từ cả câu qua self-attention. Pretraining bằng MLM buộc mọi vị trí phải đoán được từ bị che, nên vector ở mọi vị trí đều mang thông tin, không riêng [CLS].

Hai task chỉ khác nhau ở cách đọc $H$:

- **Sequence classification (LAB 08):** lấy vector [CLS], đưa qua Linear $d \to C$, được một nhãn cho cả câu.
- **Token classification (LAB 09):** áp cùng một Linear lên mọi token, $Z = HW + b$ có kích thước $T \times C$, được một nhãn BIO cho mỗi token.

Head chỉ có 6,921 tham số, trên tổng 65 triệu. Gần như toàn bộ kiến thức nằm trong encoder dùng chung. Ngoài head, chỉ cần đổi cách định nghĩa nhãn: NER gán nhãn theo word, nên phải giải quyết alignment giữa word và subword. Lab chọn chỉ gán nhãn cho token đầu của mỗi word, các subword sau nhận IGNORE.

Kết quả ủng hộ cách nhìn này. Sau 3 epoch, DistilBERT đạt entity F1 0.894, so với 0.774 của baseline dùng feature thủ công. Lợi thế nằm đúng ở chỗ cần ngữ cảnh: entity chưa gặp (recall 0.86 so với 0.66), và những từ đổi loại theo câu như "Santa Fe", lúc là địa danh lúc là công ty (đúng 67% so với 6%).

Đổi head không giải quyết được mọi thứ. Tokenizer vẫn là giới hạn chung: headline in hoa bị tách vụn và recall giảm từ 0.92 xuống 0.81. Cách đánh giá cũng phải đổi theo task: token accuracy 97.9% nghe rất cao, nhưng 83% token là O, nên phải đo ở mức entity.

Bài học: pretrained Transformer là một **backbone** tạo biểu diễn contextual. Task do head và cách định nghĩa output quyết định: một vector cho sequence task, mọi vector cho token task, và ở LAB 10 là vị trí start/end cho span task.

## Individual learning check

1. **Sequence khác token classification?** Sequence: cả câu một nhãn. Token: mỗi token một nhãn.
2. **BIO dùng để làm gì?** Mã hóa cả loại lẫn ranh giới entity bằng nhãn cho từng token.
3. **B-PER khác I-PER?** B-PER là từ đầu tên người, I-PER là các từ tiếp theo.
4. **Vì sao có label alignment problem?** Nhãn gán theo word, nhưng tokenizer tách word thành nhiều subword.
5. **IGNORE nghĩa là gì?** Token đó không tính vào loss, nhưng vẫn tham gia attention.
6. **Vì sao không chỉ dùng token accuracy?** 83% token là O, nên đoán toàn O cũng được 83%.
7. **Entity-level khác token-level?** Entity-level chỉ tính đúng khi đúng cả loại lẫn toàn bộ ranh giới.
8. **Vì sao boundary khó?** Tên dài chứa từ thường, hai entity có thể đứng liền nhau.
9. **Vì sao contextual representation có ích?** Cùng một từ có thể khác loại tùy câu ("Barcelona" là thành phố hay đội bóng).
10. **NER dùng thế nào trong IE?** Tìm entity trong văn bản thô rồi gom thành dữ liệu có cấu trúc (JSON).

## Sang LAB 10

NER tìm các entity trong câu. Question Answering tìm đoạn văn bản trả lời một câu hỏi. Backbone vẫn giữ nguyên, chỉ đổi head thành hai phân phối $P_{start}$ và $P_{end}$.

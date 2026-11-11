# Reflection

## Vì sao trong thực tế người ta thường không train Transformer từ đầu?

**Data.** Transformer có nhiều tham số và không có sẵn giả định nào về ngôn ngữ, nên cần rất nhiều dữ liệu có nhãn. Ở lab này, cùng một kiến trúc BERT-mini 11.2M tham số, model train từ đầu với 2,000 review chỉ đạt F1 0.131: nó gần như đoán toàn bộ là negative. Model pretrained với cùng 2,000 review đạt 0.788. Dữ liệu có nhãn thường đắt và ít, còn văn bản không nhãn thì gần như vô hạn.

**Pretrained representation.** Pretraining bằng MLM buộc model đoán từ bị che từ ngữ cảnh. Qua đó model học nghĩa của từ, cú pháp và quan hệ giữa các từ mà không cần nhãn. Trong notebook tokenization, BERT-mini chưa fine-tune vẫn đoán "eats" cho "The cat [MASK] fish". Frozen encoder cộng một lớp Linear 514 tham số đã đạt F1 0.74, chứng tỏ biểu diễn này mang nhiều thông tin dùng được.

**Computational cost.** Pretraining rất tốn kém, nhưng chỉ cần làm một lần rồi chia sẻ cho mọi người. Fine-tuning BERT-mini trên CPU mất khoảng 36 phút, gần bằng thời gian train from scratch, nhưng cho kết quả tốt hơn. Với 25% dữ liệu, model pretrained đạt mức mà from scratch cần gần 100% dữ liệu mới có. Chi phí còn phụ thuộc vào sequence length theo $O(T^2)$: tăng từ 128 lên 256 token làm F1 tăng 4.7 điểm nhưng thời gian tăng 2.25 lần.

**Downstream task.** Cùng một encoder có thể dùng cho nhiều task. Chỉ cần thay classification head: một head trên [CLS] cho sentiment, một head trên từng token cho NER. Phần lớn kiến thức nằm trong encoder dùng chung, còn head chỉ là phần nhỏ, riêng cho từng task.

**Fine-tuning.** Fine-tuning không bắt đầu từ trọng số ngẫu nhiên. Nó điều chỉnh toàn bộ encoder theo task, nên tốt hơn frozen hơn 10 điểm F1: MLM không dạy phân biệt positive và negative, và fine-tuning dồn thông tin đó vào [CLS]. Model pretrained cũng hội tụ nhanh hơn, đạt val_acc 0.838 ngay sau epoch đầu.

**Limitation và domain shift.** Pretrained không có nghĩa là hiểu mọi thứ. Khi đem model fine-tune trên IMDb sang SST-2, Amazon và Yelp, F1 đều giảm, nhiều nhất trên SST-2 vì câu ngắn, khác style. Model vẫn sai rất tự tin (conf > 0.99) ở các câu mỉa mai hoặc khen chê lẫn lộn, và 25.6% lỗi có confidence ≥ 0.9. Tokenizer và max_length cũng giới hạn model: 87% review bị cắt ở 128 token. Ngoài ra, fine-tuning quá lâu có thể làm model quên kiến thức chung (catastrophic forgetting).

Vì vậy, cách làm thông thường là chọn một model pretrained phù hợp với domain và fine-tune nó. Train Transformer từ đầu chỉ hợp lý khi có rất nhiều dữ liệu, hoặc domain quá khác (ngôn ngữ hay ký hiệu riêng).

## Individual learning check

1. **Pretraining khác fine-tuning ở đâu?** Pretraining học biểu diễn chung từ văn bản lớn không nhãn (MLM). Fine-tuning tiếp tục train model đó trên dữ liệu có nhãn của một task cụ thể.
2. **MLM là gì?** Che khoảng 15% token, rồi cho model đoán lại token bị che từ ngữ cảnh hai phía. Loss là $-\log P(\text{token đúng})$.
3. **Tại sao cần tokenizer?** Model chỉ nhận số. Tokenizer cắt text thành token theo đúng vocabulary của model rồi đổi sang id.
4. **Subword token giải quyết vấn đề gì?** Từ hiếm hoặc từ mới được ghép từ các mảnh nhỏ ("un ##watch ##able") thay vì thành [UNK], mà vocabulary vẫn nhỏ (30K).
5. **[CLS] dùng như thế nào?** Đây là token đặt ở đầu sequence. Vector cuối của nó được dùng làm biểu diễn của cả câu để đưa vào classification head.
6. **Attention mask dùng để làm gì?** Đánh dấu token thật (1) và padding (0), để attention bỏ qua padding.
7. **Frozen encoder nghĩa là gì?** Không cập nhật trọng số encoder, chỉ train head. Encoder đóng vai trò bộ trích feature.
8. **Fine-tuning thay đổi những tham số nào?** Thường là tất cả: embedding, các layer attention và FFN, cùng head mới.
9. **Vì sao pretrained hữu ích khi data nhỏ?** Biểu diễn ngôn ngữ đã có sẵn, nên data chỉ cần dùng để học ranh giới task (0.788 so với 0.131 ở 10% data).
10. **Pretrained có tránh được domain shift không?** Không. F1 giảm trên cả SST-2, Amazon và Yelp.
11. **Vì sao tăng sequence length làm attention tốn hơn?** Ma trận attention có kích thước $T \times T$. $T$ tăng gấp đôi thì bộ nhớ và phép tính attention tăng gấp bốn.
12. **Confidence cao có đảm bảo đúng không?** Không. Nhóm conf ≥ 0.9 vẫn sai 6.2%, và có những lỗi với conf 0.997.

## Sang LAB 09

Classification chỉ gán một nhãn cho cả câu. LAB 09 chuyển sang token classification (NER): thay vì dùng vector [CLS], ta gắn head lên vector của từng token để nhận ra ví dụ "Barack Obama" là PERSON và "Paris" là LOCATION. Khi đó, việc WordPiece tách một từ thành nhiều subword sẽ quan trọng, vì phải gán nhãn cho đúng từng từ.

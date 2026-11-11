# Error analysis

Model: BERT-mini fine-tune (max_length 128) trên toàn bộ test IMDb (24,678 review).

## Tổng quan

| | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| Pretrained (fine-tune) | 0.842 | 0.845 | 0.841 | **0.843** | 1,919 | 1,970 |
| Frozen encoder | 0.740 | 0.752 | 0.721 | 0.736 | 2,950 | 3,459 |
| From scratch | 0.815 | 0.824 | 0.805 | 0.814 | 2,135 | 2,424 |

- Lỗi của pretrained cân bằng giữa FP và FN. Frozen và from scratch thì nhầm positive thành negative nhiều hơn.
- Tỉ lệ lỗi là 16.7% ở review dài hơn 128 token, so với 9.5% ở review ngắn hơn (87% review bị cắt).

## Ví dụ lỗi

Các mẫu dưới đây là những lỗi có confidence cao nhất. Ở cả 8 mẫu, model from scratch cũng sai cùng chiều.

| # | Text (rút gọn) | True | Pred | Conf | Error type | Hypothesis |
|---|---|---|---|---|---|---|
| 4020 ★ | "one of the best Kung fu movies in the history of Cinema ... Chuck is a great actor ... a masterpiece!" | neg | pos | 0.997 | sarcasm / label ambiguity | Có thể là lời khen mỉa mai phim Chuck Norris. Câu chữ hoàn toàn tích cực nên model không có tín hiệu nào để nhận ra. |
| 3469 | "the surfing scenes are still the best ever. A true classic that can be seen many times" | neg | pos | 0.996 | label ambiguity | Văn bản chỉ có lời khen, không có ý chê. Nhãn negative nhiều khả năng là nhiễu. |
| 11859 | "This movie was pure genius ... I give it 9.5/10. Rent it today!" | neg | pos | 0.996 | label ambiguity | Có cả điểm 9.5/10 nên gần chắc là nhãn sai. Lỗi này nằm ở dữ liệu chứ không ở model. |
| 6341 | "This show is awesome ... I don't agree with the homosexual content ... throwing the balance way off" | neg | pos | 0.994 | ambiguous sentiment / long-range | Câu mở đầu khen, phần sau chê. Model bám vào "awesome" ở đầu, và review dài 150 token nên phần cuối bị cắt. |
| 20691 ★ | "It's not Citizen Kane, but it does deliver ... Badly acted and directed, poorly scripted. Who cares?" | pos | neg | 0.995 | sarcasm / context | Người viết thích phim dù thừa nhận phim dở. Model đếm các từ chê ("badly", "poorly") mà không hiểu "Who cares?" lật ngược ý. |
| 22010 | "what a piece of crap. I'm still confused ... Milo isn't worth it ... My score: 10" | pos | neg | 0.995 | label ambiguity / truncation | Nội dung chê nhưng điểm là 10, có thể là mỉa mai. Review dài 142 token nên câu "My score: 10" bị cắt, model không thấy. |
| 13467 | "Good, boring or bad? It's good ... This could have been a poorly-executed Reservoir Dogs ripoff, but the skill of the cast ..." | pos | neg | 0.994 | negation / context | Các từ chê xuất hiện trong câu hỏi tu từ và trong giả định "could have been". Model không xử lý được cấu trúc này, và phần "but the skill of the cast" bị cắt. |
| 12735 | "very talented actors ... Unfortunately, an irrational plot, and very poor writing ..." | pos | neg | 0.993 | label ambiguity | Nội dung chủ yếu chê, chỉ khen diễn viên. Model đọc đúng nội dung, nên nhiều khả năng nhãn bị lệch. |

★ là 2 mẫu model dự đoán rất tự tin nhưng sai (FP và FN có confidence cao nhất). Danh sách gồm 4 FP (#4020, 3469, 11859, 6341) và 4 FN (#20691, 22010, 13467, 12735).

**Phân loại:** 4 trên 8 lỗi nghiêng về **label ambiguity**, tức nhãn khó hoặc có thể sai. Các lỗi còn lại là **sarcasm**, **negation/context** và **truncation**. Ở mọi lỗi, model dựa vào từ khoá sentiment bề mặt.

## Câu hỏi bắt buộc

**Q1. Model nào hưởng lợi nhiều hơn khi training data ít?**
- Observation: ở 10% data, F1 của pretrained là 0.788, còn from scratch là 0.131.
- Evidence: khoảng cách là 3.3, 4.2, 9.5 và 65.7 điểm ở 100%, 50%, 25% và 10% data. Pretrained với 10% data ngang from scratch với 50%.
- Interpretation: pretrained hưởng lợi nhiều hơn, vì đã có sẵn biểu diễn ngôn ngữ và chỉ cần dữ liệu để học ranh giới phân loại.

**Q2. Pretrained tốt hơn from scratch chứng minh điều gì?**
- Observation: hơn 2.9 điểm F1 với cùng kiến trúc, dữ liệu và số epoch.
- Interpretation: chỉ chứng minh rằng khởi tạo từ pretraining tốt hơn khởi tạo ngẫu nhiên **trong thiết lập này**. Kết quả không chứng minh model "hiểu ngôn ngữ": lỗi ở trên cho thấy model vẫn dựa vào từ khoá. Kết quả cũng chưa tách được đóng góp của dữ liệu pretraining với số bước train.

**Q3. Nếu frozen gần bằng fine-tuned thì sao?**
Khi đó biểu diễn pretrained đã chứa sẵn thông tin task cần, nên có thể dùng frozen để tiết kiệm chi phí. Ở lab này điều đó **không xảy ra**: frozen kém hơn 10.7 điểm.

**Q4. Fine-tuning tốt hơn frozen rất nhiều gợi ý gì?**
- Evidence: 0.843 so với 0.736. Mean pooling (0.745) chỉ nhỉnh hơn [CLS] một chút.
- Interpretation: mục tiêu MLM không tạo ra một chiều "sentiment" tách được bằng một lớp Linear. Encoder cần được điều chỉnh để đưa thông tin sentiment vào [CLS].

**Q5. Performance giảm mạnh trên domain khác vì sao?**
- Evidence: SST-2 giảm 4.3 điểm dù cũng là review phim. Amazon giảm 1.2 và Yelp giảm 2.1.
- Interpretation: các nguyên nhân có thể là style và độ dài khác (câu ngắn so với review dài), từ vựng riêng của domain ("delicious", "battery"), hoặc phân bố nhãn khác. Head chỉ được fine-tune trên review phim.

**Q6. Sequence length tăng mà performance gần như không đổi thì có nên tăng tiếp?**
Không nên. Chi phí attention tăng theo $T^2$ mà không đổi lại được gì. Ở lab này, tăng từ 128 lên 256 vẫn có lợi (+4.7 F1, thời gian ×2.25) vì 87% review dài hơn 128 token. Nhưng từ 256 lên 512 chỉ giữ thêm được 27% review, trong khi bộ nhớ attention tăng 4 lần. Cần đo trước rồi mới quyết định.

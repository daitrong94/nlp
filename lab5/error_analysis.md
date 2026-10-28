# Error analysis

Model phân tích: GRU (max_length 200, giữ token đầu) trên test IMDb (24,678 review). Để biết token nào ảnh hưởng tới prediction, dùng **occlusion**: bỏ từng token và đo P(pos) thay đổi bao nhiêu (Δ > 0: token đẩy về positive).

## Confusion matrix

| Model | TP | FN | FP | TN | Sai trên positive | Sai trên negative |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 11,039 | 1,373 | 1,344 | 10,922 | 11.1% | 11.0% |
| RNN | 9,301 | 3,111 | 3,060 | 9,206 | 25.1% | 24.9% |
| LSTM | 10,369 | 2,043 | 2,286 | 9,980 | 16.5% | 18.6% |
| GRU | 10,910 | 1,502 | 2,218 | 10,048 | 12.1% | 18.1% |

- GRU nhầm negative thành positive nhiều hơn (FP > FN), nên recall 0.879 > precision 0.831.
- Review bị truncate có tỉ lệ lỗi 19.5%, review không bị truncate chỉ 12.1%.
- Accuracy của GRU giảm theo độ dài: 0.882 (≤ 50 token) xuống 0.758 (> 400 token). TF-IDF giữ khoảng 0.89 ở mọi độ dài.

## Correctly classified

| # | Text (rút gọn) | True | Pred | P(pos) | Analysis |
|---|---|---|---|---|---|
| 21881 | "I haven't expected that this movie to be this good ... You will definitely enjoy it! 7/10" | pos | pos | 0.998 | Tín hiệu tích cực rõ, kết thúc bằng "7/10". |
| 11121 | "I liked Buffy and Angel ... but this is too much. The worst lead actress EVER ... Crap. Crap. Crap." | neg | neg | 0.001 | Không bị câu khen ở đầu đánh lừa, phần sau "but" quyết định. |
| 2831 | "not quite bad enough to make it enjoyable ... just boring ... Dull, dull, dull, dull, dull." | neg | neg | 0.001 | Có cả "bad" lẫn "enjoyable" nhưng đọc đúng ý chê. |

## Incorrectly classified

| # | Text (rút gọn) | True | Pred | P(pos) GRU / TF-IDF | Nguyên nhân | Do sequence modeling? |
|---|---|---|---|---|---|---|
| 14830 | "Thought it was a lame ass movie, **but not at all** ... Worth watching" | pos | neg | 0.013 / 0.480 | contrast, negation | Có |
| 3827 | "one of the weaker ... lacks the usual fun ... avoid this one till near the end." | neg | pos | 0.985 / 0.204 | phần cuối lấn át | Có |
| 23571 | 398 token, kết luận "I thoroughly enjoyed this film" nằm sau token 200 | pos | neg | 0.005 / 0.189 | truncation | Có |
| 6412 | "Masterpiece ... 10/10 ... don't miss this instant classic." (phim của Carrot Top) | neg | pos | 0.984 / 0.964 | sarcasm | Không |
| 11859 | "pure genius ... I give it 9.5/10. Rent it today!" | neg | pos | 0.996 / 0.996 | nhãn sai | Không |

1. **14830:** "lame" có Δ = −0.076, lớn nhất review. GRU đọc tới "but not at all" nhưng không đảo được nghĩa của "lame". TF-IDF gần đúng hơn vì cộng đều các từ tích cực.
2. **3827:** review 62 token, toàn lời chê. Nhưng câu cuối "till near the end" đẩy về positive ("end" +0.007, "near" +0.004), còn "avoid", "poor" chỉ khoảng −0.002. GRU nghiêng về những gì đọc sau cùng, lỗi mà bag-of-words không mắc.
3. **23571:** 200 token đầu chỉ tả cốt truyện kỳ dị, kết luận khen bị cắt. Giữ token cuối thay vì token đầu sẽ giúp (LSTM tail 100 có F1 0.838, cao hơn head 200 với 0.827).
4. **6412, 11859:** mỉa mai và nhãn sai, mọi model đều sai với P > 0.94. Không thể sửa bằng cách đổi model.

## Kết luận

- **Lỗi do sequence modeling:** không đảo được nghĩa sau "but not", phần cuối lấn át phần đầu.
- **Lỗi do truncation:** nguyên nhân lớn nhất trên IMDb (19.5% so với 12.1%).
- **Lỗi không do model:** sarcasm, nhãn nhiễu.
- Với câu ngắn có contrast ("expected bad, but it was actually wonderful"), GRU đúng còn TF-IDF sai. Vấn đề chính là một hidden state 128 chiều phải tóm tắt cả review dài.

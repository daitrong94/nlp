# Error analysis

Model phân tích: Model A, TF-IDF unigram + Logistic Regression (C = 4), trên test IMDb sau khi bỏ duplicate (24,678 review). Positive class = positive. Số liệu lấy từ mục 8 - 10 của notebook, hình ở confusion_matrix.png.

## Confusion matrix

| Model | TP | FN | FP | TN | Tỉ lệ sai trên positive | Tỉ lệ sai trên negative |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 11,039 | 1,373 | 1,344 | 10,922 | 11.1% | 11.0% |
| Avg Word2Vec + LR | 10,548 | 1,864 | 1,598 | 10,668 | 15.0% | 13.0% |
| TF-IDF + Linear SVM | 11,052 | 1,360 | 1,354 | 10,912 | 11.0% | 11.0% |
| TF-IDF (uni+bigram) + LR | 11,297 | 1,115 | 1,190 | 11,076 | 9.0% | 9.7% |

- Class bị nhầm nhiều nhất: với Model A hai class bị nhầm gần như bằng nhau (positive 11.1%, negative 11.0%). Với Avg Word2Vec, positive bị nhầm nhiều hơn rõ rệt (15.0% so với 13.0%). Với uni+bigram thì negative bị nhầm nhiều hơn.
- Loại error chiếm nhiều: Model A có FN (1,373) nhỉnh hơn FP (1,344). Avg Word2Vec có FN nhiều hơn FP 266 review. Uni+bigram ngược lại, FP (1,190) nhiều hơn FN (1,115).
- Metric phản ánh: FN nhiều thì recall thấp hơn precision (Avg Word2Vec: recall 0.850, precision 0.868). FP nhiều thì precision thấp hơn recall (uni+bigram: precision 0.905, recall 0.910). Model A cân bằng nên precision 0.891 và recall 0.889 gần bằng nhau.

So sánh review đúng và sai của Model A: độ dài trung vị gần như nhau (172 và 177 token), tỉ lệ có từ phủ định cũng gần nhau (82.7% và 83.8%). Khác biệt rõ nhất là độ tự tin: trung bình max(p, 1 - p) là 0.87 ở review đúng và 0.68 ở review sai, tức phần lớn lỗi là các review có tín hiệu lẫn lộn.

## Correctly classified

| # | Text (rút gọn) | True | Predicted | P(pos) | Analysis |
|---|---|---|---|---|---|
| 24091 | "ANOTHER great performance by Kiefer Sutherland. I love his movies ... I'll be buying the dvd or tape for sure. 9/10." | positive | positive | 0.9997 | Nhiều từ positive mạnh (great, love, well) và điểm 9/10. |
| 19894 | "Excellent story of lives that need repair ... Glenn Close was excellent in the title role. It was also nice to see Christopher Walken ..." | positive | positive | 0.9996 | excellent xuất hiện hai lần, đây là feature positive mạnh thứ 2 (weight 8.12). Review ngắn nhưng tín hiệu rõ. |
| 3941 | "This was a very disappointing movie. I would definitely call this the worst movie of all time. The acting and writing were poor ... just plain awful. Avoid this wretched piece of garbage at all costs!" | negative | negative | 0.00001 | Chứa nhiều feature negative mạnh nhất: worst (-12.29), awful, poor, avoid, disappointing. |

## Incorrectly classified

| # | Text (rút gọn) | True | Predicted | P(pos) A / B / uni+bigram | Nguyên nhân |
|---|---|---|---|---|---|
| 20691 | "It's not Citizen Kane, but it does deliver. Cleavage, and lots of it. Badly acted and directed, poorly scripted. Who cares? I didn't watch it for the dialog." | positive | negative | 0.053 / 0.002 / 0.257 | negation, short text |
| 17840 | "If you're looking for a not-so-serious mob movie ... I expected Pfeiffer was poorly miscast, however, she plays her mob wife role to the hilt. Not a bad performance from Baldwin, either ..." | positive | negative | 0.019 / 0.091 / 0.134 | negation |
| 20687 | "This HAS to be my guilty pleasure ... This move has no meat, not substance, no deep thought provoking scenes. Just plain old college kids having fun ... relax and NOT think. Viva la nudity!" | positive | negative | 0.031 / 0.092 / 0.072 | negation, lexical ambiguity |
| 298 | "This is, without a doubt, the most hilarious movie I've ever seen. Seriously, if the makers of this movie are ever discovered, they'll put guys like Jim Carrey out of a job. Rent "Jack-O" tonight! Believe me, you won't regret it!" | negative | positive | 0.977 / 0.556 / 0.904 | sarcasm |
| 11859 | "This movie was pure genius. John Waters is brilliant. It is hilarious ... The acting is great ... this is probably my favorite. I give it 9.5/10. Rent it today!" | negative | positive | 0.996 / 0.997 / 0.996 | data noise |
| 12735 | "David Morse and Andre Braugher are very talented actors ... Unfortunately, an irrational plot, and very poor writing is making it difficult for me ..." | positive | negative | 0.018 / 0.316 / 0.024 | data noise |

1. Review 20691 - negation, short text
   - Người viết chê diễn xuất và kịch bản nhưng kết luận "Who cares?", tức vẫn thích phim. Ý nghĩa nằm ở "not Citizen Kane, but it does deliver" và "who cares", không nằm ở từng từ.
   - Model chỉ thấy poorly (-1.56), badly (-1.10), not, didn't. Review chỉ có 28 token nên vài từ âm đủ quyết định kết quả. Uni+bigram tăng P(pos) lên 0.257 nhưng vẫn sai.
2. Review 17840 - negation
   - Các cụm "not-so-serious", "poorly miscast" (là dự đoán ban đầu, sau đó bị phủ nhận), "not a bad performance" đều mang nghĩa trung tính hoặc tích cực.
   - Model unigram tách thành các từ riêng lẻ: ba đóng góp âm lớn nhất là poorly (-0.96), bad (-0.74), miscast (-0.49), not cũng đóng góp âm (-0.31).
3. Review 20687 - negation, lexical ambiguity
   - "no meat, not substance, no deep thought" và "NOT think" ở đây là lời khen kiểu phim giải trí. "guilty pleasure" là cách nói tích cực nhưng từ guilty riêng lẻ không mang nghĩa đó.
   - Đóng góp âm lớn nhất là no (-0.63), not (-0.34), nudity, offended. Model coi mọi từ phủ định là tín hiệu negative.
4. Review 298 - sarcasm
   - "most hilarious movie I've ever seen", "put Jim Carrey out of a job", "you won't regret it" đều là mỉa mai. Jack-O là phim kinh dị kinh phí thấp, hilarious ở đây nghĩa là dở đến buồn cười.
   - Model dựa vào hilarious, carrey, job. Avg Word2Vec ít tự tin hơn (0.556), nhưng không có representation nào ở đây hiểu được mỉa mai.
5. Review 11859 - data noise
   - Nội dung khen hoàn toàn, có cả điểm "9.5/10", nhưng label là negative. Đây là lỗi nhãn của dataset, cả ba model đều dự đoán positive với P > 0.99.
   - Trong 6 false positive tự tin nhất có 3 review thuộc dạng này (11859, 4020, 2647), nên một phần lỗi FP thực chất là nhiễu nhãn.
6. Review 12735 - data noise
   - Review chê kịch bản (unfortunately, very poor writing) và chỉ khen diễn viên, nhưng label là positive. Nhãn IMDb lấy từ điểm số của người viết (positive khi ≥ 7) nên có review mà chữ và điểm không khớp nhau.
   - Model dự đoán theo nội dung chữ là hợp lý. Không nên tối ưu model để "sửa" các lỗi kiểu này.

Ba nguyên nhân hay gặp nhất là negation, sarcasm và data noise. Negation là lỗi do representation: unigram TF-IDF và average embedding đều coi not là một từ độc lập, không biết nó phủ định từ nào. Bigram sửa được một phần (P(pos) của 20691 tăng từ 0.053 lên 0.257), nhưng không đủ khi phủ định đứng xa từ cảm xúc.

## Inspect model weights

Top features của Model A (TF-IDF unigram + LR):

| Top positive | Weight | Top negative | Weight |
|---|---|---|---|
| great | 9.86 | worst | -12.29 |
| excellent | 8.12 | bad | -10.15 |
| 7 | 7.98 | waste | -9.24 |
| perfect | 7.23 | awful | -8.93 |
| best | 6.32 | boring | -7.88 |
| wonderful | 6.24 | poor | -6.95 |
| 8 | 5.62 | poorly | -6.88 |
| amazing | 5.57 | disappointment | -6.80 |
| favorite | 5.25 | worse | -6.47 |
| brilliant | 5.13 | dull | -6.06 |
| well | 5.02 | terrible | -6.03 |
| superb | 4.99 | nothing | -5.97 |
| highly | 4.91 | avoid | -5.91 |
| loved | 4.89 | fails | -5.80 |
| love | 4.82 | unfortunately | -5.80 |
| enjoyed | 4.79 | save | -5.66 |
| enjoyable | 4.71 | annoying | -5.53 |
| today | 4.66 | horrible | -5.42 |
| fun | 4.63 | no | -5.36 |
| rare | 4.62 | lame | -5.23 |

Những feature này có ý nghĩa với task không:

- Phần lớn có: great, excellent, perfect, amazing đối lập với worst, bad, waste, awful, boring. Đây là đúng các từ con người dùng để đánh giá phim.
- Một số feature có ý nghĩa nhưng phụ thuộc domain/cách viết của IMDb: 7 và 8 (weight 7.98, 5.62) đến từ điểm tự chấm như "7/10", còn 1, 2, 3, 4 có weight âm (-4.69, -2.16, -4.21, -4.76). Đây là feature rất mạnh nhưng chỉ tồn tại ở dataset này. 10 chỉ có weight 0.84 vì tokenizer tách "1/10" thành 1 và 10, nên 10 xuất hiện ở cả review tốt lẫn xấu.
- save và today là feature theo khuôn câu: "save your money", "rent it today". Bản thân từ không mang cảm xúc.
- no, nothing, unfortunately có weight âm vì hay đi cùng lời chê. Đây là lý do model đoán sai các câu như "not bad at all" hoặc "no meat, not substance" trong review khen.
- Với uni+bigram, các cụm như the worst (-11.73), the best (+7.89), very good (+4.57), not good (-2.37), not bad (+1.59) trở thành feature riêng, giúp model phân biệt được một phần phủ định.

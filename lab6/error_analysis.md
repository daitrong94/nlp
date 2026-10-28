# Error analysis

Model phân tích: **Self-Attention** (Embedding → single-head self-attention → attention pooling → Linear; không có positional embedding). Đây là model attention có validation F1 cao hơn. Đánh giá trên test IMDb sau khi bỏ duplicate (24,678 review), max_length 200, positive class = positive. Số liệu lấy từ mục 9 - 11 của notebook, hình ở `confusion_matrix.png` và `application_attention.png`.

Với mỗi review, mình xem ba thứ:

$$
\beta_t \;(\text{pooling weight}), \qquad
(\beta A)_t = \sum_i \beta_i A_{it} \;(\text{attention rollout}), \qquad
\Delta z_t = z(\text{review}) - z(\text{review bỏ token } t) \;(\text{occlusion})
$$

- $\beta_t$ là trọng số của hàng $h_t$ khi pooling. $h_t$ đã trộn value của nhiều token, nên $\beta_t$ cao chưa chắc nghĩa là token $t$ quan trọng.
- $(\beta A)_t$ là phần của context vector đến từ value của token input $t$.
- $\Delta z_t$ tính trên logit $z$ (không phải trên P(pos), vì model rất tự tin và P bão hoà ở 0 hoặc 1). $\Delta z_t > 0$ nghĩa là token $t$ đẩy về positive. Đây là thước đo hành vi thật của model, dùng để kiểm tra các giả thuyết dựa trên attention.

## Confusion matrix

| Model | TP | FN | FP | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 11,039 | 1,373 | 1,344 | 10,922 | 0.892 | 0.889 | 0.890 |
| LSTM | 10,369 | 2,043 | 2,286 | 9,980 | 0.819 | 0.835 | 0.827 |
| **Self-Attention** | 10,765 | 1,647 | 2,386 | 9,880 | 0.819 | 0.867 | 0.842 |
| Self-Attention + pos | 10,371 | 2,041 | 2,088 | 10,178 | 0.832 | 0.836 | 0.834 |

- Self-Attention nhầm **negative** nhiều hơn: 19.5% review negative bị đoán positive, so với 13.3% review positive bị đoán negative. FP nhiều hơn FN, nên recall cao hơn precision.
- Tỉ lệ lỗi là 16.3%: 19.9% trên review bị truncate (dài hơn 200 token), 13.9% trên review không bị truncate.
- So với LSTM: 2,468 review cả hai cùng sai, 1,565 review chỉ Self-Attention sai, 1,861 review chỉ LSTM sai.

## Attention có trùng với token quan trọng không?

Trên 300 review test ngẫu nhiên (≤ 200 token):

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| $H(\beta)/\ln n$ | 0.997 | β gần như đều, pooling gần như mean pooling |
| $H(A_i)/\ln n$ trung bình | 0.714 | từng hàng của A tập trung hơn nhiều |
| Spearman$(\beta, \lvert\Delta z\rvert)$ | −0.03 | không có quan hệ thứ hạng |
| Spearman$(\beta A, \lvert\Delta z\rvert)$ | 0.35 | tương quan vừa phải |
| Top-1 trùng occlusion: β / βA | 11.7% / 52.7% | |
| $\lvert\Delta z\rvert$ khi bỏ 3 token: top β / top βA / top $\lvert\Delta z\rvert$ / ngẫu nhiên | 1.24 / 1.29 / 1.25 / 0.25 | bỏ token có attention cao làm logit đổi gấp khoảng 5 lần bỏ ngẫu nhiên |

Nhìn β trên hình gần như không thấy gì, nhưng thứ hạng ở đầu vẫn mang tín hiệu. βA gần với hành vi của model hơn hẳn. Câu hỏi "attention có nhìn đúng token không" vì vậy phụ thuộc vào việc nhìn attention nào, và chỉ trả lời được khi đối chiếu với occlusion.

## Correctly classified

| # | Text (rút gọn) | True | Pred | P(pos), z | Top βA | Top occlusion Δz | Nhận xét |
|---|---|---|---|---|---|---|---|
| 14683 | "I found this movie to be a simple yet wonderful comedy. This movie is purely entertaining. I can watch it time and time again and still enjoy the dialog ..." | positive | positive | 1.000, z = 12.1 | wonderful 0.32, is 0.10 | wonderful +5.62, purely −2.55, enjoy +0.63 | Attention và occlusion cùng chỉ vào "wonderful", đúng từ mình dự đoán là quan trọng. "entertaining" không lọt top. "purely" lại đẩy về negative (−2.55), nhưng không đủ đổi kết quả. |
| 3874 | "This is a terrible movie, don't waste your money on it. Don't even watch it for free. That's all I have to say." | negative | negative | 0.000, z = −14.3 | waste 0.34, terrible 0.13 | terrible −2.99, waste −1.87, say +1.07 | Đúng hai từ con người cũng chọn. Review ngắn, tín hiệu rõ. |
| 10512 | "Sure it is a new take on vampires. Who cares. ... This was not entertaining. It was a dull story, poorly acted, with annoying cinematography. Save your ..." | negative | negative | 0.000, z = −14.4 | save 0.19, annoying 0.11, dull 0.09, poorly 0.08 | save −3.03, dull −2.35, annoying −1.84, poorly −1.36 | dull, poorly, annoying đúng như dự đoán. Token mạnh nhất lại là "save" (trong "save your money"), một khuôn câu của IMDb mà LAB 04 cũng đã thấy. "not entertaining" không nằm trong top: model đúng mà không cần hiểu phủ định. |

Ghi chú: cũng có review đúng vì lý do khó giải thích. Review 18286 ("Rating-10 Classic Waters! One of his best ...") có βA cao nhất ở **is** (0.18) và **the** (0.16), và occlusion cho "is" +1.16, lớn hơn cả "best" (+0.99). Giả thuyết: trong self-attention, hàng của một từ chức năng có thể chép value của từ cảm xúc ("is" chú ý vào "best"/"superb"), nên bỏ "is" là bỏ một bản sao của tín hiệu đó. Token input mà attention chỉ vào không nhất thiết là nguồn của thông tin.

## Incorrectly classified

| # | Text (rút gọn) | True | Pred | P(pos) SA / LSTM / TF-IDF | Top βA | Top occlusion Δz | Nguyên nhân |
|---|---|---|---|---|---|---|---|
| 250 | "Although the story is good ... my DVD copy contained almost unbearable synchronization problems. ... I would therefore be very careful in purchasing any DVD ... I can recommend it to prospective viewers" | negative | positive | 0.999 / 0.948 / 0.884 | prospective 0.16, recommended 0.04, unbearable 0.04 | prospective +1.72, unbearable −1.07, 3 −0.90 | ambiguous wording, rare word |
| 16332 | "I love Gheorghe Muresan, so I automatically loved this movie. Everything else about it was so-so... Billy Crystal is a good actor, even if he is annoying. But the thing that made this movie was ... seeing Gheorghe Muresan act." | positive | negative | 0.002 / 0.105 / 0.047 | annoying 0.15, love 0.07, thing 0.06 | annoying −2.10, love +1.93, loved +0.98 | ambiguous wording (khen chê lẫn lộn, nhượng bộ) |
| 18672 | "What about DJ Cash Money??? This film fails in part by not covering the mid to late 80s. There was only a small mention of DJ Cheese in 86. Also, it's Grandmixer "DST", not "DXT"!!!!!" | positive | negative | 0.005 / 0.098 / 0.096 | fails 0.16, mid 0.12 | fails −3.59, mid +1.58, money −1.44 | noise (chữ và nhãn không khớp), domain |
| 3469 | "... the surfing scenes are still the best ever. A true classic that can be seen many times. ..." | negative | positive | 1.000 / 0.929 / 0.991 | great 0.17, is 0.06 | great +1.70, rides −0.57, true +0.56 | noise (nhãn sai) |
| app | "I thought this movie would be bad, but it was actually excellent." | positive | negative | 0.041 / 0.661 / 0.551 | excellent 0.27, bad 0.14 | excellent +4.22, bad −4.12, would −1.57 | contrast, long-range / word order |
| app | "The cast is wonderful, but the story is boring and the ending is awful." | negative | positive | 0.997 / 0.086 / 0.033 | wonderful 0.40, boring 0.23 | wonderful +15.97, boring −8.99, awful −4.83 | một từ mạnh lấn át, contrast |
| app | "I like this movie." | positive | negative | 0.000 / 0.567 / 0.300 | movie 0.49, this 0.22, like 0.16 | movie +2.93, i +2.80, this −2.69 | domain shift (độ dài) |

Với mỗi lỗi, trước khi xem attention mình ghi token mà mình cho là quan trọng, sau đó đặt giả thuyết và đối chiếu với occlusion. Không suy ra "attention nhìn sai nên model sai".

**1. Review 250 - ambiguous wording, rare word**

- Mình dự đoán token quan trọng là: unbearable, problems, careful (chê bản DVD) và recommend (khen phim).
- βA và occlusion cùng xếp **"prospective"** đứng đầu (βA 0.16, Δz +1.72), một từ hiếm không mang cảm xúc. Như vậy attention **không** tập trung vào token mình dự đoán.
- Giả thuyết: "prospective" có liên hệ giả với positive trong train (khuôn "recommend it to prospective viewers"). Occlusion ủng hộ giả thuyết: bỏ token này thì z giảm 1.72. Nhưng z vẫn còn khoảng 5.8, tức vẫn positive, nên "prospective" không phải nguyên nhân duy nhất. Bản thân review cũng kết thúc bằng "I can recommend it", nên nhãn negative (do điểm số thấp vì lỗi DVD) khó suy ra từ chữ. Cả LSTM và TF-IDF cũng sai.

**2. Review 16332 - ambiguous wording / nhượng bộ (không phải do attention)**

- Mình dự đoán token quan trọng là: love, loved ("I love Gheorghe Muresan, so I automatically loved this movie") và câu kết "the thing that made this movie was ... seeing Gheorghe Muresan act".
- Attention tập trung vào **"annoying"** (βA 0.15), là từ nằm trong mệnh đề nhượng bộ "even if he is annoying", tức ít quan trọng với con người. love chỉ nhận 0.07.
- Giả thuyết: model không biểu diễn được phạm vi của "even if" và "so-so" vì không có thứ tự, nên "annoying" được tính như một lời chê đầy đủ. Occlusion: bỏ "annoying" thì z tăng 2.10, nhưng vẫn còn khoảng −4.4, vẫn negative. Như vậy "annoying" chỉ là một phần: "so-so", "at least" cũng đẩy về negative. Đây là review khen chê lẫn lộn, LSTM (0.105) và TF-IDF (0.047) cũng sai. Không thể nói model sai **vì** attention vào "annoying".

**3. Review 18672 - noise (attention đúng chỗ, nhãn khó hiểu)**

- Mình dự đoán token quan trọng là "fails" (lời chê duy nhất có cảm xúc rõ).
- Attention đúng như vậy: fails có βA 0.16 và Δz −3.59. Review chỉ toàn bắt lỗi chi tiết (thiếu DJ Cash Money, sai tên "DST"), không có câu khen nào, nhưng nhãn là positive (người viết chấm ≥ 7).
- Model đoán theo nội dung chữ là hợp lý. Đây là lỗi do dữ liệu (chữ và nhãn không khớp) và kiến thức domain (hip-hop), không phải do sequence modeling hay attention. Review 3469 thuộc cùng nhóm theo chiều ngược lại: khen hoàn toàn ("the best ever", "a true classic"), nhãn negative, mọi model đều đoán positive với P > 0.92.

**4. "I thought this movie would be bad, but it was actually excellent." - contrast / thứ tự (do representation)**

- Mình dự đoán token quan trọng là "excellent" và vai trò đảo ý của "but".
- Attention có nhìn vào excellent (βA 0.27) và bad (0.14). Occlusion: excellent +4.22, bad −4.12, hai lực gần như triệt tiêu, cộng với bias nghiêng negative của các từ còn lại ("would" −1.57) nên ra negative.
- Đây là lỗi do representation chứ không do attention: Self-Attention không có vị trí, nên "I thought this movie would be excellent, but it was actually bad" có **cùng** bag-of-words và cho **cùng** prediction. Model không thể biết vế nào đứng sau "but". LSTM (0.661) đoán đúng. Ở mục 7.1, Self-Attention + pos phân biệt được loại cặp này ("expected bad but it was great" 0.896 so với "expected great but it was bad" 0.026). Như vậy thông tin vị trí là thứ còn thiếu.

**5. "The cast is wonderful, but the story is boring and the ending is awful." - một từ mạnh lấn át**

- Mình dự đoán token quan trọng là boring và awful (hai lời chê về phần chính) cùng "but".
- Attention dồn vào "wonderful" (βA 0.40). Heatmap cho thấy nhiều hàng (the, is, cast) cùng chú ý vào "wonderful", nên value của nó được chép vào nhiều vị trí. Vì β gần đều, context gần như là trung bình các hàng, nên càng nhiều hàng chứa "wonderful" thì nó càng thắng. Occlusion xác nhận: wonderful +15.97, lớn hơn boring (−8.99) và awful (−4.83) cộng lại.
- Giả thuyết (khớp với occlusion): cơ chế "nhiều query cùng chọn một key mạnh" làm một từ cảm xúc rất mạnh quyết định cả câu, bất kể vị trí và "but". LSTM đúng (0.086).

**6. "I like this movie." - domain shift (độ dài)**

- Mình dự đoán token quan trọng là "like".
- Attention dồn vào **movie** (βA 0.49), like chỉ 0.16. Model đoán negative với z = −11.0, tức tự tin cực độ.
- Giả thuyết: model chưa từng thấy input 4 token (chỉ 2.4% review train ≤ 50 token, review ngắn nhất 10 token). Với rất ít token, context gần như chỉ là value của một hai từ, và "this movie" hay xuất hiện trong review chê (TF-IDF cũng chỉ cho 0.300). Occlusion cho thấy không token nào đủ kéo z về gần 0 (|Δz| ≤ 2.93 so với z = −11). Như vậy lỗi đến từ phân phối độ dài khác train, không phải từ một token bị nhìn sai. "I do not like this movie." được đoán đúng (negative), nhưng cũng với z = −12.2, nên mình không coi đó là bằng chứng model hiểu phủ định.

## Kết luận

- **Lỗi do representation của self-attention không vị trí** (contrast, "but", negation): model là một bag-of-words phi tuyến, nên không phân biệt được vế trước và vế sau. Thêm vị trí giúp được một phần (mục 7.1).
- **Lỗi do cơ chế attention:** khi nhiều query cùng chọn một key mạnh, một từ cảm xúc lấn át cả câu; với input rất ngắn, context chỉ còn một hai value nên model tự tin quá mức.
- **Lỗi không phụ thuộc model:** nhãn nhiễu (3469, 18672), khen chê lẫn lộn (16332), truncation (19.9% lỗi trên review bị cắt). Attention trong các trường hợp này thường chỉ đúng vào từ con người cũng chọn.
- Attention cho biết model dựa vào token nào (βA khớp occlusion khoảng một nửa số lần), nhưng không giải thích vì sao model sai. Mọi kết luận ở trên là giả thuyết đã được đối chiếu với occlusion, không phải hệ quả trực tiếp của heatmap.

# Error analysis

Model phân tích: **GRU** (validation F1 cao nhất trong ba sequence model), max_length = 200 (giữ 200 token đầu), trên test IMDb sau khi bỏ duplicate (24,678 review). Positive class = positive. Số liệu lấy từ mục 9 - 11 của notebook, hình ở `confusion_matrix.png`.

Để giải thích từng prediction, notebook dùng **occlusion**: bỏ lần lượt từng token và đo

$$
\Delta_i = P(\text{pos} \mid \text{review}) - P(\text{pos} \mid \text{review bỏ token } i)
$$

$\Delta_i > 0$ nghĩa là token $i$ đẩy prediction về positive, $\Delta_i < 0$ là đẩy về negative. Khác với weight của Logistic Regression ở LAB 04, $\Delta_i$ phụ thuộc vào context: cùng một từ có thể có $\Delta$ khác dấu ở hai câu khác nhau.

## Confusion matrix

| Model | TP | FN | FP | TN | Tỉ lệ sai trên positive | Tỉ lệ sai trên negative |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 11,039 | 1,373 | 1,344 | 10,922 | 11.1% | 11.0% |
| RNN | 9,301 | 3,111 | 3,060 | 9,206 | 25.1% | 24.9% |
| LSTM | 10,369 | 2,043 | 2,286 | 9,980 | 16.5% | 18.6% |
| GRU | 10,910 | 1,502 | 2,218 | 10,048 | 12.1% | 18.1% |

- Class bị nhầm nhiều nhất: với GRU là **negative** (18.1% bị đoán thành positive, so với 12.1% theo chiều ngược lại). LSTM cũng nhầm negative nhiều hơn. RNN và TF-IDF nhầm hai class gần như bằng nhau.
- Loại lỗi chiếm nhiều: GRU có FP (2,218) nhiều hơn FN (1,502), nên recall 0.879 cao hơn precision 0.831.
- So với TF-IDF: 2,017 review GRU sai nhưng TF-IDF đúng, 1,014 review ngược lại, 1,703 review cả hai cùng sai. Hai model sai ở những chỗ khá khác nhau.

## Review sai khác review đúng ở đâu

| | n | Median độ dài | Bị truncate | Có phủ định | Có từ đối lập (but, however, ...) | Độ tự tin trung bình |
|---|---|---|---|---|---|---|
| GRU sai | 3,720 | 210 token | 52.1% | 87.7% | 85.3% | 0.784 |
| GRU đúng | 20,958 | 168 token | 38.3% | 83.5% | 78.0% | 0.917 |

- Tỉ lệ lỗi trên review bị truncate (dài hơn 200 token) là 1,937 / 9,955 = **19.5%**, trên review không bị truncate là 1,783 / 14,723 = **12.1%**.
- Theo độ dài: accuracy của GRU là 0.882 (≤ 50 token), 0.880 (101 - 200), 0.827 (201 - 400), 0.758 (> 400). TF-IDF giữ khoảng 0.89 ở mọi nhóm. Phần lớn khoảng cách giữa GRU và TF-IDF nằm ở review dài.
- Review có từ đối lập chiếm tỉ lệ cao hơn trong nhóm sai (85.3% so với 78.0%), đúng với kiểu lỗi "khen trước, chê sau" hoặc ngược lại.

## Correctly classified

| # | Text (rút gọn) | True | Predicted | P(pos) GRU / TF-IDF | Analysis |
|---|---|---|---|---|---|
| 21881 | "I have to say that I haven't expected that this movie to be this good. ... The acting is truthfully and not exaggerated ... Watch it! You will definitely enjoy it! 7/10" | positive | positive | 0.998 / 0.973 | Tín hiệu tích cực rải đều và kết thúc bằng "definitely enjoy it! 7/10". Occlusion: 7 và 10 ở cuối có Δ dương lớn nhất. "not exaggerated" không bị hiểu thành tiêu cực. |
| 11121 | "Dude...I liked Buffy and Angel as much as the next sci-fi freak...but this is too much. The worst lead actress EVER!! ... you should NEVER waste your time watching this show ... Crap. Crap. Crap." | negative | negative | 0.001 / 0.003 | "I liked Buffy and Angel" ở đầu là khen chương trình khác, sau "but" là chê. Model không bị câu khen đầu tiên đánh lừa. Các từ âm mạnh nhất: worst, crap, waste. |
| 2831 | "Yet again not quite bad enough to make it enjoyable. In fact this one is just boring. ... (Apart from James Mason who is always great ...) ... Dull, dull, dull, dull, dull." | negative | negative | 0.001 / 0.004 | "not quite bad enough to make it enjoyable" có cả bad lẫn enjoyable nhưng nghĩa là chê. Câu "great" nằm trong ngoặc về một diễn viên. Kết thúc bằng 5 lần "dull" nên hidden state cuối rõ ràng negative. |

## Incorrectly classified

| # | Text (rút gọn) | True | Predicted | P(pos) GRU / LSTM / RNN / TF-IDF | Nguyên nhân | Do sequence modeling? |
|---|---|---|---|---|---|---|
| 14830 | "Just want to inform you guys that this movie was actually pretty good !!! Thought it was a lame ass movie, but not at all, many moments in this movie wore pretty horrifying. ... Worth watching if u like a kick ass horror movie thats for sure!!" | positive | negative | 0.013 / 0.090 / 0.225 / 0.480 | contrast, negation, domain term | **Có** |
| 3827 | "This is definitely one of the weaker of the series of Carry On films. It lacks the usual fun and sparkle and even the cast seem embarrassed by the poor dialogue. ... avoid this one till near the end." | negative | positive | 0.985 / 0.878 / 0.871 / 0.204 | recency / long-range | **Có** |
| 13517 | "Walter Matthau and Jack Lemmon, both of whom are sadly missed, proved once again that they were a team dedicated to ... bringing hilarious moments to the screen. ... a further tribute to their illustrious careers." (181 token) | positive | negative | 0.024 / 0.062 / 0.343 / 0.594 | ambiguous / từ đa nghĩa | Một phần |
| 23571 | "I want to preface this review by saying that I have no idea what Begotten is truly about ... [sau token 200:] overall I would have to say that I thoroughly enjoyed this film ..." (398 token) | positive | negative | 0.005 / - / - / 0.189 | truncation, domain | **Có** (truncation) |
| 19553 | "This movie made me laugh so hard that it hurt ... this movie was intended to be bad and cheezy ... corny dialogue, bad one liners and horrible special effects. And it's all made to deliver laughter ..." (219 token) | positive | negative | 0.004 / - / - / 0.160 | sarcasm / "so bad it's good" | Không |
| 6412 | "Masterpiece. Carrot Top blows the screen away. Never has one movie captured the essence of the human spirit quite like "Chairman of the Board." 10/10... don't miss this instant classic." | negative | positive | 0.984 / 0.944 / 0.944 / 0.964 | sarcasm | Không |
| 11859 | "This movie was pure genius. John Waters is brilliant. ... The acting is great ... this is probably my favorite. I give it 9.5/10. Rent it today!" | negative | positive | 0.996 / 0.990 / 0.956 / 0.996 | noise (nhãn sai) | Không |

**1. Review 14830 - contrast + negation (lỗi do sequence modeling)**

- Câu then chốt là "Thought it was a lame ass movie, **but not at all**". Muốn hiểu đúng, model phải dùng "but not at all" ở phía sau để phủ định "lame" ở phía trước. Đây đúng là loại dependency mà sequence model được kỳ vọng xử lý.
- Occlusion: "lame" có $\Delta = -0.076$, lớn nhất trong review. "not", "all", "good", "worth" chỉ đóng góp khoảng $+0.001$ đến $+0.006$. GRU đã đọc tới "but not at all" nhưng không đảo được nghĩa của "lame". TF-IDF lại gần đúng (0.480) vì cộng đều các từ tích cực.
- Ngoài ra review có nhiều từ không chuẩn ("wore" thay cho "were", "u", "kinda", "kick ass") mà với horror fan thì "horrifying" và "blood, gore" là lời khen (domain-specific).

**2. Review 3827 - recency (lỗi do sequence modeling)**

- Review chỉ 62 token nên không bị truncate. Mọi từ đánh giá đều tiêu cực: weaker, lacks, embarrassed, poor, terminal decline, avoid. TF-IDF đoán đúng (0.204).
- GRU lại đoán positive với P = 0.985, và cả LSTM (0.878) lẫn RNN (0.871) cũng sai theo cùng hướng. Câu cuối "avoid this one till near the end" là lời khuyên xem các phần khác trước. Model đọc trái sang phải, nên h_T bị chi phối bởi phần cuối "... till near the end", và occlusion cho "end" ($+0.007$), "near" ($+0.004$) là hai token đẩy về positive mạnh nhất, trong khi "avoid", "poor", "lacks" chỉ khoảng $-0.002$.
- Khi P bão hoà ở 0.985, bỏ một token không đổi được kết quả nhiều, nên $\Delta$ nhỏ. Nhưng xu hướng chung khớp với Experiment 6: thông tin ở đầu bị phần cuối lấn át. Đây là lỗi mà bag-of-words không mắc.

**3. Review 13517 - từ đa nghĩa, ambiguous sentiment (một phần do sequence)**

- "sadly missed" nói về hai diễn viên đã mất, không phải chê phim. "Too bad that fellow dance instructors ... are given so little to do" là chê nhẹ một chi tiết. Toàn bài là lời khen (hilarious hai lần, riotous, lovelier, tribute).
- Occlusion: "sadly" có $\Delta = -0.179$, lớn nhất review. "example" ($-0.073$) và "just" ($-0.070$) trong "just another example of this" cũng đẩy về negative, có lẽ vì "just another ..." hay xuất hiện trong review chê. "hilarious" gần như không đóng góp.
- TF-IDF đúng (0.594) vì có nhiều từ tích cực được cộng lại. GRU để vài cụm ở đầu quyết định trạng thái, các lời khen phía sau không kéo lại được.

**4. Review 23571 - truncation (lỗi do cách đưa sequence vào model)**

- Review dài 398 token. 200 token đầu chỉ mô tả nội dung kỳ dị của phim ("god kills himself", "impregnate herself") và nói "I have no idea what Begotten is truly about", nên rất dễ hiểu thành chê.
- Kết luận nằm ở cuối, đã bị cắt: "there were many scenes that were in fact beautiful ... overall I would have to say that I thoroughly enjoyed this film ... not a bad thing". Model không bao giờ thấy câu này.
- TF-IDF thấy cả bài nhưng cũng sai (0.189), vì phần mô tả quá dài và tiêu cực về từ vựng. Với GRU, truncation là nguyên nhân trực tiếp: nếu giữ phần cuối (tail), model có cơ hội đọc được kết luận. Ở mục 7, LSTM tail 100 có F1 0.838, cao hơn head 200 (0.827).

**5. Review 19553 - "so bad it's good" (không phải lỗi do sequence modeling)**

- Người viết khen phim **vì** nó dở: "intended to be bad and cheezy", "corny dialogue, bad one liners and horrible special effects ... all made to deliver laughter". Từ vựng gần như toàn tiêu cực nhưng ý là khen.
- Cả GRU (0.004) và TF-IDF (0.160) đều sai. Hiểu được cần kiến thức về thể loại phim B-movie và ý định của người viết, không chỉ là thứ tự từ.

**6. Review 6412 - sarcasm (không phải lỗi do sequence modeling)**

- "Masterpiece", "captured the essence of the human spirit", "10/10", "instant classic" về một phim của Carrot Top. Mỉa mai chỉ nhận ra được nếu biết phim này nổi tiếng là dở.
- Mọi model đều đoán positive với P > 0.94. Không có tín hiệu nào trong văn bản mà thứ tự từ có thể khai thác.

**7. Review 11859 - nhãn nhiễu (noise)**

- Review khen hoàn toàn, tự chấm 9.5/10, nhưng label là negative. Đây chính là review đã xuất hiện trong error analysis của LAB 04. Mọi model đều đoán positive với P > 0.95.
- Trong 6 false positive tự tin nhất của GRU có ít nhất 3 review dạng này (11859, 2647, 4020). Một phần lỗi FP thực chất là nhiễu nhãn, không nên tối ưu model để "sửa" chúng.

## Kết luận

- **Lỗi do sequence modeling** (14830, 3827, một phần 13517): model có thông tin trong input nhưng không dùng đúng. Có thể do không đảo được nghĩa sau "but not at all", để phần cuối lấn át phần đầu, hoặc để một cụm sớm quyết định toàn bộ. Đây là các lỗi TF-IDF không mắc, và là lý do GRU sai ở 2,017 review mà TF-IDF đúng.
- **Lỗi do truncation** (23571): thông tin quyết định không bao giờ vào model. Đây là nguyên nhân lớn nhất trên IMDb (lỗi 19.5% so với 12.1%).
- **Lỗi không phụ thuộc representation** (19553, 6412, 11859): sarcasm, "so bad it's good", nhãn nhiễu. Mọi model đều sai, cần kiến thức ngoài văn bản hoặc dữ liệu sạch hơn.
- Không thể kết luận đơn giản là "GRU không hiểu câu". Ở các câu ngắn có contrast ("I expected the movie to be bad, but it was actually wonderful"), GRU đúng còn TF-IDF sai (mục 11). Vấn đề là hidden state kích thước cố định phải tóm tắt cả review dài, và nghiêng về những gì đọc sau cùng.

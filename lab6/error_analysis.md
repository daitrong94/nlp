# Error analysis

Model phân tích: Self-Attention (không có positional embedding), max_length 200, test IMDb (24,678 review).

Với mỗi review, so sánh hai thứ:

- **Attention (βA):** phần của context vector đến từ mỗi token.
- **Occlusion (Δz):** bỏ token đó thì logit thay đổi bao nhiêu (Δz > 0: token đẩy về positive). Đây là ảnh hưởng thật của token, dùng để kiểm tra attention.

## Confusion matrix

| Model | TP | FN | FP | TN | F1 |
|---|---|---|---|---|---|
| TF-IDF + LR | 11,039 | 1,373 | 1,344 | 10,922 | 0.890 |
| LSTM | 10,369 | 2,043 | 2,286 | 9,980 | 0.827 |
| Self-Attention | 10,765 | 1,647 | 2,386 | 9,880 | 0.842 |
| Self-Attention + pos | 10,371 | 2,041 | 2,088 | 10,178 | 0.834 |

- Self-Attention nhầm negative thành positive nhiều hơn (FP > FN).
- Tỉ lệ lỗi: 19.9% trên review bị truncate, 13.9% trên review không bị truncate.

## Attention có trùng với token quan trọng không?

Trên 300 review test:

- Pooling weight β gần như đều, không tương quan với occlusion (Spearman −0.03).
- Attention rollout βA tương quan vừa phải (0.35). Token có βA cao nhất trùng với occlusion trong 52.7% review.
- Bỏ 3 token có attention cao nhất làm logit đổi khoảng 1.3, gấp 5 lần bỏ 3 token ngẫu nhiên (0.25).

Attention có liên quan tới prediction, nhưng không phải là lời giải thích chính xác.

## Correctly classified

| # | Text (rút gọn) | True | Pred | Top attention | Top occlusion | Nhận xét |
|---|---|---|---|---|---|---|
| 14683 | "a simple yet wonderful comedy ... purely entertaining ... still enjoy the dialog" | pos | pos | wonderful 0.32 | wonderful +5.6 | Attention đúng từ quan trọng |
| 3874 | "This is a terrible movie, don't waste your money on it." | neg | neg | waste 0.34, terrible 0.13 | terrible −3.0, waste −1.9 | Đúng từ con người cũng chọn |
| 10512 | "not entertaining ... dull story, poorly acted, annoying cinematography. Save your ..." | neg | neg | save 0.19, annoying 0.11 | save −3.0, dull −2.4 | "save" (save your money) là khuôn câu quen thuộc của IMDb |

## Incorrectly classified

| # | Text (rút gọn) | True | Pred | P(pos) Attn / LSTM | Top attention | Top occlusion | Nguyên nhân |
|---|---|---|---|---|---|---|---|
| 250 | "story is good ... unbearable synchronization problems ... I can recommend it to prospective viewers" | neg | pos | 0.999 / 0.948 | prospective 0.16 | prospective +1.7 | rare word, ambiguous |
| 16332 | "I love Gheorghe Muresan, so I automatically loved this movie ... Billy Crystal is a good actor, even if he is annoying" | pos | neg | 0.002 / 0.105 | annoying 0.15 | annoying −2.1, love +1.9 | ambiguous wording |
| 18672 | "This film fails in part by not covering the mid to late 80s ..." | pos | neg | 0.005 / 0.098 | fails 0.16 | fails −3.6 | noise (chữ và nhãn không khớp) |
| app | "I thought this movie would be bad, but it was actually excellent." | pos | neg | 0.041 / 0.661 | excellent 0.27, bad 0.14 | excellent +4.2, bad −4.1 | contrast, không có thứ tự |
| app | "The cast is wonderful, but the story is boring and the ending is awful." | neg | pos | 0.997 / 0.086 | wonderful 0.40 | wonderful +16.0 | một từ mạnh lấn át |

Mỗi lỗi dưới đây là một **giả thuyết**, đã được đối chiếu với occlusion, không suy ra trực tiếp từ heatmap.

1. **250 - rare word:** attention dồn vào "prospective", một từ không mang cảm xúc. Occlusion xác nhận từ này đẩy về positive (+1.7). Nhưng bỏ nó thì logit vẫn dương, nên đây không phải nguyên nhân duy nhất. Bản thân review cũng kết thúc bằng "I can recommend it".
2. **16332 - khen chê lẫn lộn:** attention dồn vào "annoying", từ nằm trong mệnh đề nhượng bộ "even if he is annoying". Model không có thứ tự nên không hiểu phạm vi của "even if". Bỏ "annoying" thì logit vẫn âm, vì "so-so" cũng góp phần.
3. **18672 - nhãn nhiễu:** attention đúng vào "fails", review toàn bắt lỗi nhưng nhãn lại là positive. Model đoán theo chữ là hợp lý.
4. **"would be bad, but it was actually excellent":** "excellent" và "bad" gần như triệt tiêu nhau. Model không có vị trí nên đổi chỗ hai từ cũng cho cùng kết quả. LSTM đoán đúng.
5. **"wonderful, but ... boring ... awful":** nhiều token cùng chú ý vào "wonderful", nên từ này lấn át cả boring lẫn awful. LSTM đoán đúng.

Ngoài ra, "I like this movie." bị đoán negative với P = 0.000. Model chưa từng thấy câu ngắn như vậy (chỉ 2.4% review train ≤ 50 token), nên đây là lỗi domain shift.

## Kết luận

- **Lỗi do không có thứ tự:** contrast, "but", phủ định. Thêm positional embedding giúp được một phần.
- **Lỗi do cơ chế attention:** một từ mạnh lấn át cả câu, tự tin quá mức với câu ngắn.
- **Lỗi không do model:** nhãn nhiễu, khen chê lẫn lộn, truncation.
- Attention cho biết model dựa vào token nào, nhưng không giải thích vì sao model sai.

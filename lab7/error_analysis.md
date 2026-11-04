# Error analysis

Model phân tích: Transformer (1 block, 4 head, PE, mean pooling, max_length 128) trên test IMDb (24,678 review).

Để kiểm tra attention, so sánh hai thứ:

- **Attention:** trọng số trung bình mà mỗi token nhận được (trung bình trên các head và các hàng).
- **Occlusion (Δz):** bỏ token đó thì logit thay đổi bao nhiêu (Δz > 0: token đẩy về positive).

## Confusion matrix

| Model | Precision | Recall | F1 |
|---|---|---|---|
| LSTM | 0.794 | 0.842 | 0.818 |
| Transformer | 0.833 | 0.790 | 0.811 |
| Transformer (no PE) | 0.817 | 0.818 | 0.818 |
| Transformer CLS | 0.823 | 0.763 | 0.792 |

- Transformer nhầm positive thành negative nhiều hơn (precision > recall). LSTM thì ngược lại.
- Tỉ lệ lỗi: 20.1% trên review bị truncate (> 128 token), 14.3% trên review không bị truncate.
- Attention và occlusion tương quan vừa phải (Spearman 0.27). Token có attention cao nhất trùng với token quan trọng nhất trong 48.5% review.

## Correctly classified

| # | Text (rút gọn) | True | Pred | Top attention | Top occlusion |
|---|---|---|---|---|---|
| 14127 | "A delightful and wonderful film ... The cast is excellent" | pos | pos (0.999) | wonderful, delightful, excellent | wonderful, delightful |
| 17967 | "A wonderful film - a charming, quirky family story ... a really nice movie" | pos | pos (0.999) | wonderful, family, charming | wonderful, charming |
| 19143 | "This is a delightful film ... does a wonderful job ... makes you feel happy" | pos | pos (0.999) | wonderful, delightful, easily | delightful, wonderful |

Với review khen rõ ràng, attention và occlusion cùng chỉ vào đúng các từ cảm xúc mạnh.

## Incorrectly classified

| # | Text (rút gọn) | True | Pred | Top attention | Top occlusion | Nguyên nhân |
|---|---|---|---|---|---|---|
| 17840 | "a not-so-serious mob movie ... I expected Pfeiffer was poorly miscast, however, she plays ..." | pos | neg (0.006) | poorly, looking, bad | poorly −1.44, bad −0.42 | negation, contrast |
| 6412 | "Masterpiece. Carrot Top blows the screen away ... 10/10 ... instant classic." | neg | pos (0.997) | carrot, masterpiece, 10 | masterpiece +1.34 | sarcasm |
| 4020 | "one of the best Kung fu movies ... Chuck is a great actor ... a masterpiece" | neg | pos (0.998) | great, best, masterpiece | great +0.45 | noise (nhãn sai) |
| 21696 | "looks like it's gonna cost you £198.00 ... IMDB won't let me submit ... blahblahblah" | pos | neg (0.006) | looks, gonna, like | looks −0.61 | noise (không có nội dung review) |
| app | "I thought this movie would be bad, but it was actually excellent." | pos | neg (0.129) | bad, thought, excellent | bad −4.83, excellent +2.78 | contrast |

1. **17840:** "poorly miscast" là điều người viết **từng nghĩ** rồi phủ nhận bằng "however". Attention và occlusion cùng dồn vào "poorly". Giả thuyết: model không hiểu được phạm vi của "expected ... however". Occlusion ủng hộ giả thuyết này, và LSTM cũng sai.
2. **6412:** mỉa mai. Attention nhìn đúng "masterpiece", nhưng hiểu được ý mỉa mai cần biết phim này nổi tiếng là dở. Mọi model đều sai.
3. **4020, 21696:** lỗi dữ liệu. Review 4020 khen hoàn toàn nhưng nhãn negative. Review 21696 không nói gì về phim. Model đoán theo chữ là hợp lý.
4. **Contrast:** "bad" (−4.83) lấn át "excellent" (+2.78), dù "excellent" nằm sau "but". Có PE nhưng model chưa học được vai trò của "but" từ 20K review.

## Kết luận

- Attention thường chỉ vào từ cảm xúc mạnh, kể cả khi model sai. Nó cho biết model **nhìn vào đâu**, không cho biết model **hiểu đúng hay không**.
- Lỗi do model: contrast, phủ định có phạm vi xa ("expected ... however").
- Lỗi không do model: sarcasm, nhãn sai, review không có nội dung.

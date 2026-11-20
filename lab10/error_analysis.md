# Error analysis

Model: DistilBERT fine-tune, test là SQuAD validation (10,570 câu hỏi). Có 2,681 câu sai (EM = 0).

| Loại lỗi | Số câu | F1 trung bình |
|---|---|---|
| wrong start (end đúng) | 665 | 0.62 |
| wrong end (start đúng) | 578 | 0.57 |
| wrong boundary (cả hai đầu sai, còn chồng lấn) | 111 | 0.43 |
| wrong span, cùng câu với gold | 615 | 0.06 |
| wrong span, câu khác | 712 | 0.08 |

## Wrong start

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| When did Tesla patent the motor? | in May 1888 | May 1888 | wrong start | Gold lấy cả giới từ "in", model bỏ |
| Why do these defections occur? | economic or financial factors | because of economic or ... | wrong start | Model lấy thừa "because of" |
| What has a Lama determined to do? | be reborn | reborn | wrong start | Thiếu động từ "be" |
| When did Martin Luther die? | 18 February 1546 | 10 November 1483 – 18 February 1546 | multiple plausible spans | Lấy cả ngày sinh đứng ngay trước |
| What does the IPCC not do? | carry out research nor ... monitor climate related data | monitor climate related data | partial answer | Chỉ lấy vế sau của câu phủ định kép |

## Wrong end

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| What talk show followed Super Bowl 50 on CBS? | The Late Show with Stephen Colbert | The Late Show | wrong end | Dừng trước phần bổ nghĩa của tên riêng |
| How many people ... died of plague in Baghdad? | two-thirds of its population | two-thirds | partial answer | Đúng con số, thiếu "of its population" |
| The region spans which mountains ...? | Peninsular Ranges | Peninsular Ranges with their peaks | wrong end | Lấy thừa cụm phía sau |
| How popular was Coronation Street ...? | the most popular show | the most popular | wrong end | Thiếu danh từ "show" |
| What did lane and vail finance? | an electric lighting company in Tesla's name | electric lighting company | partial answer | Mất phần "in Tesla's name" |

## Wrong span

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| Which NFL team represented the NFC at Super Bowl 50? | Carolina Panthers | Denver Broncos | distractor | Cùng câu có hai đội, model chọn đội AFC |
| Who did the Broncos play in the last week ...? | San Diego Chargers | Brock Osweiler | distractor | Chọn tên người thay vì tên đội |
| Which company aired multiple ads at a steep discount? | Anheuser-Busch InBev | beer manufacturer | multiple plausible spans | Cùng chỉ một công ty, model lấy cách gọi chung |
| What party forms the Scottish Parliament? | hold the majority of seats | Scottish Government | question ambiguity | Câu hỏi viết sai ý, gold cũng khó hiểu |
| What was Tymnet | an international data communications network ... | X.25/X.75 gateways | long context | Câu định nghĩa ở xa, model chọn chi tiết kỹ thuật gần từ "Tymnet" |

## Wrong start và wrong end

| | Số câu |
|---|---|
| #wrong start (tính cả boundary và span) | 2,103 |
| #wrong end | 2,016 |

Hai số gần bằng nhau. Start sai nhiều hơn một chút.

Giả thuyết: ở đầu span hay có từ chức năng như "in", "because of", "be" hoặc mạo từ. Người gán nhãn không thống nhất có lấy các từ này hay không, nên model khó học. Ở cuối span, lỗi chủ yếu là lấy thiếu hoặc thừa phần bổ nghĩa ("of its population", "with Stephen Colbert").

Phần lớn lỗi ranh giới vẫn có F1 khoảng 0.6, nên model tìm đúng vùng. Lỗi nặng thật sự là wrong span (F1 ≈ 0), thường xảy ra khi context có nhiều đáp án cùng loại.

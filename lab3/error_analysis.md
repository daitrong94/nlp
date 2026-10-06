# Error analysis

Model dùng để phân tích: Skip-gram, 100 chiều, window 5, train trên C4 30K document.

Cột Evidence liệt kê các context chung của hai từ, kèm số lần mỗi context xuất hiện cạnh từ thứ nhất / từ thứ hai. Số liệu lấy từ mục 5 và mục 12 của notebook.

## Similarity đúng

| Cặp | Observed | Expected | Evidence from corpus |
|---|---|---|---|
| doctor → physician | 0.728, rank 5 | rất gần (đồng nghĩa) | prescribe (4/3), consult (23/10), visits, medicine; khuôn "consult your doctor/physician" |
| doctor → dentist | 0.768, rank 2 | gần (cùng loại nghề) | consult, appointment, talk, ask; khuôn "see a dentist", "ask your doctor" |
| cat → dog | 0.784, rank 1 | gần (thú nuôi) | collars, lover, towel; cụm "cat and dog" |

Possible explanation: các cặp này thay thế được cho nhau trong cùng khuôn câu, nên có context gần như giống nhau.

## Similarity sai / bất ngờ

1. **football → replica, shirtre** (0.776, rank 1)
   - Expected: soccer, rugby.
   - Explanation: noisy data và domain bias. Corpus có các trang spam lặp lại cụm "cheap football shirts replica football shirt" (68 câu).
   - Evidence: token rác "shirtre", "thcheap" nằm trong số các context chung.
2. **hot → cold** (0.748, rank 1) và **good → bad** (0.743, rank 3)
   - Expected: thấp, vì là từ trái nghĩa.
   - Explanation: từ trái nghĩa có cùng context. Model đo mức độ liên quan, không đo cùng hay ngược nghĩa. SimLex-999 ρ chỉ đạt 0.35.
   - Evidence: 104 câu chứa "hot or cold". good/bad đi cùng luck, news, idea.
3. **hospital → upmc, samitivej** (0.764, rank 1)
   - Expected: clinic, hospitals.
   - Explanation: frequency. Đây là tên riêng hiếm, chỉ xuất hiện 7–8 lần, nên vector không đáng tin.
   - Evidence: context chung chỉ có at, to, a. Tương tự với doctor → ophthalmologist (17 lần) và banana → lentil (7 lần).

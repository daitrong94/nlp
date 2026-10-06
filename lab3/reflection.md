# Reflection

## Polysemy - bank

- Các từ gần bank nhất đều thuộc nghĩa tài chính: hsbc, barclays, citibank, banking.
- cos(bank, account) = 0.58, trong khi cos(bank, river) = 0.29.
- Trong corpus có 414 lần bank mang nghĩa tài chính, chỉ 12 lần nghĩa bờ sông.

Một vector duy nhất không thể biểu diễn cả hai nghĩa. Nó là trung bình của mọi context bank xuất hiện, nên bị nghĩa phổ biến hơn lấn át.

Context thì đủ thông tin để phân biệt: lấy trung bình word vector của các từ xung quanh từng lần xuất hiện, rồi phân loại theo centroid gần nhất, đúng 97% với nghĩa tài chính và 92% với nghĩa bờ sông.

## Bảng so sánh

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không | Sparse | Không |
| Co-occurrence | Không | Sparse (thành dense nếu dùng SVD) | Không |
| Word2Vec | Không | Dense | Không |
| Contextual embedding | Có | Dense | Có |

**Tại sao bank cần contextual representation?** Nghĩa của bank chỉ xác định được khi nhìn câu chứa nó. Contextual embedding (Transformer) tính vector cho từng lần xuất hiện dựa trên các từ xung quanh, nên "river bank" và "bank account" có vector khác nhau.

## Learning check

- **Distributional hypothesis:** từ xuất hiện trong context giống nhau thì có nghĩa gần nhau.
- **Tại sao doctor và physician gần nhau:** chúng dùng chung context như consult, prescribe, medicine (cos = 0.73). Khi corpus nhỏ (1,000 document), quan hệ này biến mất.
- **CBOW và Skip-gram khác nhau ở đâu:** CBOW dùng context để đoán từ ở giữa. Skip-gram dùng từ ở giữa để đoán context. Skip-gram chậm hơn (247 s so với 107 s) nhưng cho kết quả tốt hơn (WordSim353 0.62 so với 0.57).
- **Window lớn vừa tốt vừa xấu:** khi tăng window 2 → 10, các từ cùng chủ đề gần nhau hơn (WordSim353 0.57 → 0.64), nhưng độ đồng nghĩa giảm (doctor–physician 0.76 → 0.67), kết quả dễ bị nhiễu bởi spam và train chậm hơn.
- **Tại sao Word2Vec không phân biệt được hai nghĩa của bank:** mỗi từ chỉ có một vector, và vector này không phụ thuộc vào câu đang xét.
- **Tại sao TF-IDF không phải word embedding:** TF-IDF biểu diễn document, mỗi từ chiếm một chiều độc lập nên doctor ⟂ physician. Nó không học quan hệ giữa các từ.

## Kết luận

Pipeline context → co-occurrence → PPMI → SVD đã cho embedding gần bằng Word2Vec. Tuy vậy, mọi static embedding đều phụ thuộc mạnh vào corpus và chỉ có một vector cho mỗi từ. Đây là lý do cần contextual embedding và Transformer.

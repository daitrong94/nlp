# LAB 03 - Word Representations and Embeddings

## Files

- calculations.md: các bài tính tay
- prediction.md: dự đoán trước khi chạy và đối chiếu với kết quả
- cooccurrence.py: core implementation (vocabulary, co-occurrence matrix, cosine, most_similar, PPMI, SVD, analogy)
- word_embedding.ipynb: toàn bộ experiments
- results.csv: số liệu do notebook ghi ra
- error_analysis.md, reflection.md

## Cách chạy

```
pip install numpy scipy pandas matplotlib scikit-learn gensim jupyter
python cooccurrence.py        # unit test
```

Sau đó chạy toàn bộ word_embedding.ipynb, mất khoảng 25–30 phút.

## Thiết lập

- Corpus: C4, 30,000 document (khoảng 11 triệu token), tokenize giống LAB 02.
- Co-occurrence: vocabulary gồm 20,000 từ, ma trận lưu dạng sparse, có thêm PPMI và SVD 100 chiều.
- Word2Vec: Skip-gram, negative sampling, vector_size = 100, window = 5, min_count = 5, epochs = 10.

## Kết quả chính

| Model | WordSim353 ρ | SimLex-999 ρ | Analogy |
|---|---|---|---|
| PPMI + SVD 100 | 0.602 | 0.302 | - |
| CBOW (d = 100, w = 5) | 0.574 | 0.300 | - |
| Skip-gram (d = 100, w = 5) | 0.618 | 0.348 | 0.277 |
| Skip-gram (w = 10) | 0.638 | 0.333 | 0.279 |
| Skip-gram (d = 300) | 0.605 | 0.379 | 0.254 |

- doctor → ophthalmologist, dentist, pharmacist, veterinarian, physician.
- king − man + woman → queen.
- Lỗi điển hình: football → replica (spam), hot ↔ cold (trái nghĩa), bank chỉ mang nghĩa tài chính.

## AI assistance statement

- Tool: Claude Code 
- Purpose: viết code, chạy experiment, soạn nháp các file .md
- Generated content: cooccurrence.py, word_embedding.ipynb, results.csv, bản nháp calculations.md, prediction.md, error_analysis.md, reflection.md
- Verification: unit test trong cooccurrence.py, đối chiếu kết quả tính tay với code, đối chiếu số liệu trong các file .md với output của notebook

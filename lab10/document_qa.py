"""
LAB 10 - Mini Document QA: user question + tài liệu → QA model → answer span.

Tài liệu: một FAQ tiếng Anh về học phần, viết lại từ nội dung các file đề W8 - W10 (hạn nộp, dataset, metric,
quy định AI, rubric). Đáp án gold của 20 câu hỏi kiểm chứng được trực tiếp trên file đề.
Tài liệu dài hơn 384 token nên model phải dùng sliding window.

    python document_qa.py                         # đánh giá EM / F1 trên 20 câu hỏi
    python document_qa.py "When is the LAB 10 deadline?"
"""

import os
import sys
from pathlib import Path

from qa_model import QAModel, evaluate, score

DOCUMENT = """\
The course Natural Language Processing and Applications is taught in Semester I of 2026 at the VNU University of Science. \
The lecturer and teaching assistant is Pham Ngoc Hai. Every lab is submitted before 23:59 on its deadline day.

LAB 08 is about pretrained Transformers and BERT fine-tuning. Students fine-tune a pretrained model for sentiment \
classification on the IMDb movie review dataset and compare it with a Transformer trained from scratch. In the masked \
language modeling exercise, the model predicts the word eats with probability 0.70 for the sentence The cat [MASK] fish. \
The deadline of LAB 08 is November 11, 2026.

LAB 09 moves from sequence classification to token classification. The task is named entity recognition on the \
CoNLL-2003 dataset, which has four entity types: person, organization, location and miscellaneous. Labels follow the \
BIO scheme. The error analysis of LAB 09 needs at least five false positives, five false negatives and five boundary \
errors. The deadline of LAB 09 is November 18, 2026.

LAB 10 studies extractive question answering on SQuAD, the Stanford Question Answering Dataset. The model predicts a \
start position and an end position, and the answer is the span of the context between them. The main metrics are \
Exact Match and F1. Long contexts are split into overlapping windows with a maximum length of 384 tokens and a stride \
of 128 tokens. The error analysis of LAB 10 needs at least five wrong starts, five wrong ends and five wrong spans. \
The deadline of LAB 10 is November 25, 2026.

The LAB 10 rubric has a total of 100 points. Tokenization and span alignment is worth 15 points, the QA model \
implementation is worth 15 points, and the document QA application is worth 5 points. The reflection, which must be \
between 300 and 500 words, is also worth 5 points.

Students may use AI tools for documentation, debugging, visualization and boilerplate code. They may not ask AI to \
write the predictions, the error analysis or the reflection. Students who use AI for code must report what was \
generated, what was modified and what was verified.

After LAB 10, LAB 11 introduces semantic retrieval, which finds the documents relevant to a question in a large \
collection. LAB 12 combines retrieval and generation into retrieval-augmented generation, also called RAG."""

QUESTIONS = [
    ("Who is the lecturer of the course?", ["Pham Ngoc Hai"]),
    ("At which university is the course taught?", ["VNU University of Science"]),
    ("What time are labs due on the deadline day?", ["23:59", "before 23:59"]),
    ("Which dataset is used for sentiment classification in LAB 08?", ["IMDb", "IMDb movie review dataset"]),
    ("What probability does the model give to eats?", ["0.70"]),
    ("When is LAB 08 due?", ["November 11, 2026"]),
    ("Which dataset is used for named entity recognition?", ["CoNLL-2003", "CoNLL-2003 dataset"]),
    ("How many entity types does CoNLL-2003 have?", ["four"]),
    ("Which labeling scheme is used for entities?", ["BIO", "BIO scheme"]),
    ("When must LAB 09 be submitted?", ["November 18, 2026"]),
    ("What does SQuAD stand for?", ["Stanford Question Answering Dataset", "the Stanford Question Answering Dataset"]),
    ("What are the main metrics for question answering?", ["Exact Match and F1"]),
    ("What is the maximum window length?", ["384 tokens", "384"]),
    ("How much do neighbouring windows overlap?", ["128 tokens", "128", "a stride of 128 tokens"]),
    ("What is the deadline of LAB 10?", ["November 25, 2026"]),
    ("How many points is the whole LAB 10 rubric worth?", ["100 points", "100"]),
    ("How many points does tokenization and span alignment give?", ["15 points", "15"]),
    ("How long should the reflection be?", ["between 300 and 500 words", "300 and 500 words"]),
    ("What must students report when AI helps with their code?", ["what was generated, what was modified and what was verified"]),
    ("What technique combines retrieval and generation?", ["retrieval-augmented generation", "RAG"]),
]


def default_model_dir():
    if os.environ.get("QA_MODEL_DIR"):
        return os.environ["QA_MODEL_DIR"]
    local = Path(__file__).resolve().parent.parent / "hf_models" / "lab10_qa_model"
    return str(local) if local.is_dir() else "lab10_qa_model"


def run(model_dir=None, verbose=True):
    """Trả lời 20 câu hỏi trên DOCUMENT, trả về (metrics, bảng chi tiết)."""
    import pandas as pd
    qa = QAModel(model_dir or default_model_dir())
    preds = qa.answer_batch([q for q, _ in QUESTIONS], [DOCUMENT] * len(QUESTIONS))
    metrics, _ = evaluate([p["text"] for p in preds], [g for _, g in QUESTIONS])
    rows = []
    for (q, g), p in zip(QUESTIONS, preds):
        em, f1 = score(p["text"], g)
        rows.append({"question": q, "gold": g[0], "prediction": p["text"], "EM": em, "F1": round(f1, 2)})
    df = pd.DataFrame(rows)
    if verbose:
        n_tok = len(qa.tok(DOCUMENT)["input_ids"])
        print(f"Tài liệu: {len(DOCUMENT)} ký tự, {n_tok} token → cần sliding window (max_length 384, stride 128)")
        print(f"EM {metrics['EM']:.1f} | F1 {metrics['F1']:.1f} trên {len(QUESTIONS)} câu hỏi")
    return metrics, df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        qa = QAModel(default_model_dir())
        p = qa.answer(" ".join(sys.argv[1:]), DOCUMENT)
        print(f"Answer: {p['text']!r} (ký tự {p['start_char']} - {p['end_char']}, score {p['score']:.2f})")
    else:
        import pandas as pd
        pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
        metrics, df = run()
        print(df.to_string(index=False))

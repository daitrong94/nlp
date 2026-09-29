"""
LAB 02 - Core Implementation: N-gram Language Model.

Tự cài đặt unigram / bigram / trigram language model (không dùng thư viện LM).
Hỗ trợ:
- MLE:      P(w | h) = C(h, w) / C(h)
- Laplace:  P(w | h) = (C(h, w) + 1) / (C(h) + V)

Quy ước:
- Mỗi câu được thêm (n-1) token BOS "<s>" ở đầu và 1 token EOS "</s>" ở cuối.
- "<s>" chỉ đóng vai trò context, không bao giờ được dự đoán nên không tính vào V.
- Từ không có trong vocabulary được thay bằng "<unk>".
- V = |vocabulary| gồm các từ thường + "</s>" + "<unk>".
"""

import copy
import math
import re
from collections import Counter, defaultdict

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lowercase, giữ lại chữ/số (và dạng rút gọn như don't, it's)."""
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


def split_sentences(text):
    """Tách document thành các câu theo dấu . ! ? và xuống dòng."""
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def preprocess_documents(documents):
    """list[str] document -> list[list[str]] câu đã tokenize (bỏ câu rỗng)."""
    sentences = []
    for doc in documents:
        for sent in split_sentences(doc):
            tokens = tokenize(sent)
            if tokens:
                sentences.append(tokens)
    return sentences


# ---------------------------------------------------------------------------
# Vocabulary & counting
# ---------------------------------------------------------------------------

def build_vocabulary(sentences, min_count=1):
    """
    Trả về set các từ có tần suất >= min_count, cộng thêm EOS và UNK.
    Từ hiếm (< min_count) sẽ bị map thành UNK khi train/đánh giá.
    """
    counts = Counter(w for sent in sentences for w in sent)
    vocab = {w for w, c in counts.items() if c >= min_count}
    vocab.update([EOS, UNK])
    return vocab


def replace_oov(sentence, vocabulary):
    """Thay các từ ngoài vocabulary bằng UNK."""
    return [w if w in vocabulary else UNK for w in sentence]


def pad_sentence(sentence, n):
    """Thêm (n-1) BOS ở đầu và EOS ở cuối."""
    return [BOS] * (n - 1) + list(sentence) + [EOS]


def extract_ngrams(tokens, n):
    """Sinh các n-gram (tuple) từ một list token."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(sentences, n, vocabulary=None):
    """
    Đếm n-gram trên các câu đã tokenize (có padding BOS/EOS).
    Nếu truyền vocabulary thì từ OOV được map thành UNK trước khi đếm.
    Trả về Counter {tuple n-gram: count}.
    """
    counts = Counter()
    for sent in sentences:
        if vocabulary is not None:
            sent = replace_oov(sent, vocabulary)
        counts.update(extract_ngrams(pad_sentence(sent, n), n))
    return counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NGramLanguageModel:
    def __init__(self, n, smoothing="mle", min_count=1):
        if n < 1:
            raise ValueError("n phải >= 1")
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing phải là 'mle' hoặc 'laplace'")
        self.n = n
        self.smoothing = smoothing
        self.min_count = min_count
        self.vocabulary = set()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self._continuations = None  # context -> {word: count}, tạo khi cần
        self._context_vocab = set()  # vocabulary + BOS

    @property
    def V(self):
        return len(self.vocabulary)

    def fit(self, corpus, vocabulary=None):
        """
        corpus: list[list[str]] câu đã tokenize.
        vocabulary: dùng vocabulary có sẵn (để các model cùng V), nếu None thì
        tự build từ corpus với min_count.
        """
        self.vocabulary = set(vocabulary) if vocabulary is not None \
            else build_vocabulary(corpus, self.min_count)
        self._context_vocab = self.vocabulary | {BOS}
        self.ngram_counts = count_ngrams(corpus, self.n, self.vocabulary)

        # C(h) = sum_w C(h, w): tổng số lần h xuất hiện làm context.
        # Tính từ chính n-gram counts để phân phối P(. | h) cộng lại đúng bằng 1.
        self.context_counts = Counter()
        for ngram, c in self.ngram_counts.items():
            self.context_counts[ngram[:-1]] += c
        self._continuations = None
        return self

    def with_smoothing(self, smoothing):
        """Bản sao dùng chung counts nhưng đổi cách smoothing (không train lại)."""
        other = copy.copy(self)
        other.smoothing = smoothing
        return other

    def _get_continuations(self):
        if self._continuations is None:
            cont = defaultdict(dict)
            for ngram, c in self.ngram_counts.items():
                cont[ngram[:-1]][ngram[-1]] = c
            self._continuations = cont
        return self._continuations

    def _normalize_context(self, context):
        """Lấy (n-1) từ cuối của context, map OOV, pad BOS nếu thiếu."""
        if self.n == 1:
            return ()
        if isinstance(context, str):
            context = tokenize(context)
        context = replace_oov(list(context), self._context_vocab)
        context = [BOS] * (self.n - 1) + context
        return tuple(context[-(self.n - 1):])

    def probability(self, context, word):
        """
        P(word | context). context là str hoặc list token (chỉ dùng n-1 từ cuối).
        Với MLE, context chưa từng gặp (C(h) = 0) được coi là P = 0.
        """
        h = self._normalize_context(context)
        w = word if word in self.vocabulary else UNK
        return self._prob(h, w)

    def _prob(self, h, w):
        """P(w | h) với h là tuple (n-1) token và w đã được chuẩn hoá."""
        c_hw = self.ngram_counts.get(h + (w,), 0)
        c_h = self.context_counts.get(h, 0)
        if self.smoothing == "laplace":
            return (c_hw + 1) / (c_h + self.V)
        return c_hw / c_h if c_h > 0 else 0.0

    def _scored_ngrams(self, sentence, add_eos=True):
        """Các cặp (context, word) cần tính xác suất cho một câu."""
        if isinstance(sentence, str):
            sentence = tokenize(sentence)
        tokens = [BOS] * (self.n - 1) + replace_oov(sentence, self.vocabulary)
        if add_eos:
            tokens.append(EOS)
        start = self.n - 1
        return [(tuple(tokens[i - start:i]), tokens[i]) for i in range(start, len(tokens))]

    def sentence_probability(self, sentence, add_eos=True):
        """P(S) = prod_t P(w_t | context_t). Dễ underflow với câu dài."""
        p = 1.0
        for h, w in self._scored_ngrams(sentence, add_eos):
            p *= self._prob(h, w)
        return p

    def sentence_log_probability(self, sentence, add_eos=True):
        """log P(S) = sum_t log P(w_t | context_t). Trả về -inf nếu có P = 0."""
        logp = 0.0
        for h, w in self._scored_ngrams(sentence, add_eos):
            p = self._prob(h, w)
            if p == 0.0:
                return float("-inf")
            logp += math.log(p)
        return logp

    def evaluate(self, sentences):
        """
        Tính perplexity trên tập câu. N = số token được dự đoán (gồm cả EOS).
        Trả về dict: perplexity, n_tokens, n_zero_prob (số token có P = 0),
        zero_rate. Nếu có token P = 0 thì perplexity = inf.
        """
        total_logp = 0.0
        n_tokens = 0
        n_zero = 0
        for sent in sentences:
            for h, w in self._scored_ngrams(sent):
                n_tokens += 1
                p = self._prob(h, w)
                if p == 0.0:
                    n_zero += 1
                else:
                    total_logp += math.log(p)
        ppl = float("inf") if n_zero > 0 else math.exp(-total_logp / n_tokens)
        return {
            "perplexity": ppl,
            "n_tokens": n_tokens,
            "n_zero_prob": n_zero,
            "zero_rate": n_zero / n_tokens if n_tokens else 0.0,
        }

    def perplexity(self, sentences):
        """PP(W) = exp(-1/N * sum log P(w_i | context_i))."""
        return self.evaluate(sentences)["perplexity"]

    def next_word_distribution(self, context, top_k=None, exclude=(BOS, UNK)):
        """
        Phân phối P(w | context), sắp xếp giảm dần.
        - MLE: chỉ gồm các từ đã theo sau context trong train (rỗng nếu context
          chưa gặp).
        - Laplace: gồm toàn bộ vocabulary.
        Trả về list[(word, prob)].
        """
        h = self._normalize_context(context)
        if self.smoothing == "laplace":
            candidates = self.vocabulary
        else:
            candidates = self._get_continuations().get(h, {}).keys()
        dist = [(w, self._prob(h, w)) for w in candidates if w not in exclude]
        dist.sort(key=lambda x: -x[1])
        return dist[:top_k] if top_k else dist

    def continuation_log_probability(self, context, continuation):
        """
        log P(continuation | context): chỉ cộng log-prob các từ của continuation,
        context dùng làm lịch sử. Không thêm EOS.
        Trả về (log_prob, số token) để có thể chuẩn hoá theo độ dài.
        """
        if isinstance(context, str):
            context = tokenize(context)
        if isinstance(continuation, str):
            continuation = tokenize(continuation)
        history = list(context)
        logp = 0.0
        for w in continuation:
            p = self.probability(history, w)
            if p == 0.0:
                return float("-inf"), len(continuation)
            logp += math.log(p)
            history.append(w)
        return logp, len(continuation)

    def rank_continuations(self, context, candidates):
        """
        Xếp hạng candidate theo log P(candidate | context) giảm dần.
        Trả về list[dict] gồm candidate, log_prob, avg_log_prob (chia số token).
        """
        rows = []
        for cand in candidates:
            logp, length = self.continuation_log_probability(context, cand)
            rows.append({
                "candidate": cand,
                "log_prob": logp,
                "avg_log_prob": logp / length if length else float("-inf"),
            })
        rows.sort(key=lambda r: -r["log_prob"])
        return rows

    def count_unseen_ngrams(self, sentences):
        """Số n-gram (token-level và loại khác nhau) trong sentences chưa gặp ở train."""
        unseen_tokens = 0
        unseen_types = set()
        total = 0
        for sent in sentences:
            for h, w in self._scored_ngrams(sent):
                total += 1
                if (h + (w,)) not in self.ngram_counts:
                    unseen_tokens += 1
                    unseen_types.add(h + (w,))
        return {
            "total_ngrams": total,
            "unseen_ngrams": unseen_tokens,
            "unseen_rate": unseen_tokens / total if total else 0.0,
            "unseen_types": len(unseen_types),
        }


# ---------------------------------------------------------------------------
# Functional API theo đề bài (mục 14)
# ---------------------------------------------------------------------------

def train_unigram(corpus, smoothing="mle", vocabulary=None, min_count=1):
    return NGramLanguageModel(1, smoothing, min_count).fit(corpus, vocabulary)


def train_bigram(corpus, smoothing="mle", vocabulary=None, min_count=1):
    return NGramLanguageModel(2, smoothing, min_count).fit(corpus, vocabulary)


def train_trigram(corpus, smoothing="mle", vocabulary=None, min_count=1):
    return NGramLanguageModel(3, smoothing, min_count).fit(corpus, vocabulary)


def probability(model, context, word):
    return model.probability(context, word)


def sentence_probability(model, sentence):
    return model.sentence_probability(sentence)


def sentence_log_probability(model, sentence):
    return model.sentence_log_probability(sentence)


if __name__ == "__main__":
    # --- Unit tests trên corpus đồ chơi (khác corpus của bài tính tay) ---
    toy = [tokenize(s) for s in ["a b c", "a b d", "b c"]]
    eps = 1e-9

    vocab = build_vocabulary(toy)
    assert vocab == {"a", "b", "c", "d", EOS, UNK}

    bi_counts = count_ngrams(toy, 2)
    assert bi_counts[(BOS, "a")] == 2
    assert bi_counts[("a", "b")] == 2
    assert bi_counts[("c", EOS)] == 2
    assert sum(bi_counts.values()) == 3 + 3 + 2 + 3  # mỗi câu: len + 1 (EOS)

    # Unigram: N = 11 token (gồm 3 EOS), C(b) = 3
    uni = train_unigram(toy)
    assert abs(uni.probability([], "b") - 3 / 11) < eps
    assert abs(sum(uni.probability([], w) for w in uni.vocabulary) - 1) < eps

    # Bigram MLE: P(c | b) = C(b c) / C(b) = 2 / 3
    bi = train_bigram(toy)
    assert abs(bi.probability(["b"], "c") - 2 / 3) < eps
    assert bi.probability(["a"], "c") == 0.0
    assert abs(sum(bi.probability(["b"], w) for w in bi.vocabulary) - 1) < eps

    # Xác suất câu "a b c" = P(a|<s>) P(b|a) P(c|b) P(</s>|c) = 2/3 * 1 * 2/3 * 1
    expected = 2 / 3 * 1 * 2 / 3 * 1
    assert abs(bi.sentence_probability("a b c") - expected) < eps
    assert abs(bi.sentence_log_probability("a b c") - math.log(expected)) < eps
    assert bi.sentence_log_probability("a c") == float("-inf")

    # Laplace: V = 6, P(c | a) = (0 + 1) / (2 + 6)
    bi_lap = train_bigram(toy, smoothing="laplace")
    assert bi_lap.V == 6
    assert abs(bi_lap.probability(["a"], "c") - 1 / 8) < eps
    assert abs(sum(bi_lap.probability(["b"], w) for w in bi_lap.vocabulary) - 1) < eps
    assert bi_lap.perplexity([tokenize("a c")]) < float("inf")

    # Trigram: P(c | a b) = C(a b c) / C(a b) = 1 / 2
    tri = train_trigram(toy)
    assert abs(tri.probability(["a", "b"], "c") - 0.5) < eps
    assert abs(tri.probability(["x", "a", "b"], "c") - 0.5) < eps  # chỉ dùng 2 từ cuối

    # OOV được map thành <unk>
    assert bi_lap.probability(["zzz"], "a") == bi_lap.probability([UNK], "a")

    # Perplexity: PP = P(W)^(-1/N), N = 4 token của "a b c </s>"
    assert abs(bi.perplexity([tokenize("a b c")]) - expected ** (-1 / 4)) < eps

    # with_smoothing dùng chung counts
    assert abs(bi.with_smoothing("laplace").probability(["a"], "c") - 1 / 8) < eps
    assert bi.smoothing == "mle"

    # Next-word distribution & ranking
    assert bi.next_word_distribution("b")[0] == ("c", 2 / 3)
    ranked = bi.rank_continuations("a", ["b c", "c b"])
    assert ranked[0]["candidate"] == "b c"

    print("All tests passed.")

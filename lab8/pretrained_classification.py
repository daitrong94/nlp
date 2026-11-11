"""
LAB 08 - Pretrained Transformer (BERT-mini) cho sentiment classification.

Pipeline: text → tokenizer → input_ids, attention_mask → BERT encoder → [CLS] → Dropout → Linear → logits.

- Model: google/bert_uncased_L-4_H-256_A-4 (BERT-mini: 4 layer, hidden 256, 4 head, 11.2M tham số).
- Tokenizer: bert-base-uncased (WordPiece, cùng vocabulary 30,522 token với BERT-mini).
- Model và tokenizer được lưu ở ../hf_models (HF_HOME), load offline.

Ba chế độ:
- pretrained=True, freeze=False : fine-tuning toàn bộ
- pretrained=True, freeze=True  : feature extraction, encoder đóng băng, chỉ train head
- pretrained=False              : cùng kiến trúc, trọng số ngẫu nhiên (train from scratch)
"""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_models"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
TOKENIZER_NAME = "bert-base-uncased"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


class BertClassifier(nn.Module):
    """Encoder (pretrained hoặc ngẫu nhiên) + classification head task-specific trên [CLS]."""

    def __init__(self, pretrained=True, freeze=False, n_classes=2, dropout=0.1):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_NAME))   # khởi tạo ngẫu nhiên
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def cls_representation(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]                       # vector của [CLS]

    def forward(self, input_ids, attention_mask):
        return self.head(self.drop(self.cls_representation(input_ids, attention_mask)))


def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def encode(tokenizer, texts, max_length):
    enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict_logits(model, input_ids, attention_mask, batch_size=128):
    model.eval()
    out = []
    for i in range(0, len(input_ids), batch_size):
        ids, am = input_ids[i:i + batch_size], attention_mask[i:i + batch_size]
        L = int(am.sum(1).max())                                  # cắt padding thừa để chạy nhanh hơn
        out.append(model(ids[:, :L], am[:, :L]))
    return torch.cat(out).numpy()


def train(model, train_data, val_data, epochs=2, lr=1e-4, batch_size=32, warmup=0.1, seed=42, log=print):
    """Fine-tune với AdamW + linear warmup/decay. Giữ checkpoint có validation loss thấp nhất."""
    torch.manual_seed(seed)
    (Xtr, Mtr, ytr), (Xva, Mva, yva) = train_data, val_data
    ytr_t = torch.tensor(ytr)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(Xtr) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    gen = torch.Generator().manual_seed(seed)
    hist, best, t_start = [], None, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.perf_counter(), 0.0
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            L = int(Mtr[idx].sum(1).max())
            loss = F.cross_entropy(model(Xtr[idx, :L], Mtr[idx, :L]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        logits = predict_logits(model, Xva, Mva)
        val_loss = float(F.cross_entropy(torch.tensor(logits), torch.tensor(yva)))
        val_acc = float((logits.argmax(1) == yva).mean())
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": val_loss, "val_acc": val_acc,
                     "epoch_time_s": time.perf_counter() - t0})
        log(f"  epoch {ep}: train_loss {hist[-1]['train_loss']:.4f} val_loss {val_loss:.4f} "
            f"val_acc {val_acc:.4f} ({hist[-1]['epoch_time_s']:.0f} s)")
        if best is None or val_loss < best[0]:
            best = (val_loss, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return {"history": hist, "best_epoch": best[1], "train_time_s": time.perf_counter() - t_start}


if __name__ == "__main__":
    tok = load_tokenizer()
    enc = tok(["The cats are playing.", "A short one"], padding=True, truncation=True, return_tensors="pt")
    print("keys:", list(enc.keys()))
    assert tok.convert_ids_to_tokens(enc["input_ids"][0]) == ["[CLS]", "the", "cats", "are", "playing", ".", "[SEP]"]
    assert enc["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 0, 0]          # câu 2 ([CLS] a short one [SEP]) được pad 2 vị trí

    full, frozen, scratch = BertClassifier(), BertClassifier(freeze=True), BertClassifier(pretrained=False)
    print("params:", n_params(full), "| trainable khi freeze:", n_params(frozen, True))
    assert n_params(frozen, True) == 256 * 2 + 2                                # chỉ còn head
    assert n_params(full) == n_params(scratch)                                  # cùng kiến trúc
    pre = full.encoder.embeddings.word_embeddings.weight
    assert not torch.allclose(pre, scratch.encoder.embeddings.word_embeddings.weight)

    # padding không làm đổi output của câu ngắn
    full.eval()
    with torch.no_grad():
        a = full(enc["input_ids"][1:2, :5], enc["attention_mask"][1:2, :5])
        b = full(enc["input_ids"][1:2], enc["attention_mask"][1:2])
    assert torch.allclose(a, b, atol=1e-5)
    print("All tests passed.")

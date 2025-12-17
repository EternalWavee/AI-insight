import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x

class KeywordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)  # per-token binary logits

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) 1 for real tokens, 0 for pad
        returns: logits (batch, seq_len)
        """
        x = self.embed(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        # src_key_padding_mask expects True at positions that are PAD (i.e., to be masked)
        src_key_padding_mask = attention_mask == 0  # (batch, seq_len) bool
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch, d_model)
        out = out.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.classifier(out).squeeze(-1)  # (batch, seq_len)
        return logits

def build_vocab(sentences: List[List[str]], min_freq: int = 1):
    freq = {}
    for s in sentences:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    # special tokens: PAD -> 0, UNK -> 1
    idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in freq.items():
        if c >= min_freq and w not in idx:
            idx[w] = len(idx)
    return idx

class SeqDataset(Dataset):
    def __init__(self, sentences: List[List[str]], labels: List[List[int]], vocab: dict, max_len: int = None):
        assert len(sentences) == len(labels)
        self.vocab = vocab
        self.sentences = sentences
        self.labels = labels
        if max_len is None:
            self.max_len = max(len(s) for s in sentences)
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def encode_word(self, w):
        return self.vocab.get(w, self.vocab["<UNK>"])

    def __getitem__(self, idx):
        words = self.sentences[idx]
        labs = self.labels[idx]
        seq = [self.encode_word(w) for w in words]
        # pad
        pad_len = self.max_len - len(seq)
        seq_ids = seq + [self.vocab["<PAD>"]] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        labs_padded = labs + [0] * pad_len
        return torch.tensor(seq_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(labs_padded, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")  # sum to average manually excluding pads
    for seq_ids, masks, labs in dataloader:
        seq_ids = seq_ids.to(device)
        masks = masks.to(device)
        labs = labs.to(device)
        logits = model(seq_ids, masks)  # (batch, seq_len)
        # compute loss only on real tokens
        loss = criterion(logits, labs)
        # normalize by number of real tokens
        n_real = masks.sum().float().clamp_min(1.0)
        loss = loss / n_real

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq_ids.size(0)  # loss per sample
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for seq_ids, masks, labs in dataloader:
            seq_ids = seq_ids.to(device)
            masks = masks.to(device)
            labs = labs.to(device)
            logits = model(seq_ids, masks)
            loss = criterion(logits, labs)
            total_loss += loss.item()
            total_tokens += masks.sum().item()
    return (total_loss / total_tokens) if total_tokens > 0 else 0.0

def infer_sentence(model, vocab, text: str, device, threshold: float = 0.5):
    """
    text: space-separated words, e.g., "我 想 订 机票"
    returns: list of (word, prob, is_keyword_bool)
    """
    model.eval()
    words = text.split()
    ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
    seq = torch.tensor([ids], dtype=torch.long).to(device)  # (1, seq_len)
    mask = torch.tensor([[1] * len(ids)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(seq, mask)  # (1, seq_len)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return [(w, p, p >= threshold) for w, p in zip(words, probs)]

if __name__ == "__main__":
    sentences = [
        ["我", "想", "订", "机票"],
        ["帮", "我", "查", "一下", "天气"],
        ["明天", "晚上", "八点", "叫", "我", "起床"],
        ["把", "窗户", "打开"],
        ["提醒", "我", "明天", "开会"],
        ["帮", "我", "订", "餐厅"],
    ]
    labels = [
        [0, 0, 1, 1],            # 订 机票
        [0, 0, 1, 0, 1],         # 查 / 天气
        [1, 0, 0, 1, 0, 1],      # 明天 / 叫 / 起床
        [0, 1, 0],               # 窗户
        [0, 0, 1, 1],            # 明天 / 开会 (示例)
        [0, 0, 1, 0],            # 订 餐厅
    ]

    # build vocab
    vocab = build_vocab(sentences)
    pad_idx = vocab["<PAD>"]
    print("Vocab size:", len(vocab))

    # dataset & dataloader
    maxlen = max(len(s) for s in sentences)
    ds = SeqDataset(sentences, labels, vocab, max_len=maxlen)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeywordTransformer(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=2, pad_idx=pad_idx).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20

    for ep in range(1, epochs + 1):
        train_loss = train_epoch(model, dl, optimizer, device)
        val_loss = evaluate(model, dl, device)
        print(f"Epoch {ep}/{epochs}  train_loss={train_loss:.4f}  token_loss={val_loss:.6f}")

    examples = [
        "我 想 订 明天 的 机票",
        "明天 晚上 八点 叫 我 起床",
        "帮 我 查 天气",
    ]
    print("\n--- inference ---")
    for ex in examples:
        res = infer_sentence(model, vocab, ex, device, threshold=0.5)
        print(ex, "->", res)

    torch.save({"model_state": model.state_dict(), "vocab": vocab}, "keyword_transformer.pth")
    print("saved model -> keyword_transformer.pth")

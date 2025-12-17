import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class KeywordMLP(nn.Module):
    """
    输入：词向量（每个词对应一行）
    输出：每个词的概率（是否为关键词）
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, input_dim)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x  # (batch, 1)

class WordDataset(Dataset):
    def __init__(self, vectorizer, sentences, labels):
        """
        sentences: [["我","想","订","机票"], ...]
        labels:    [[0,0,1,1], ...]
        """
        self.samples = []
        for words, lab in zip(sentences, labels):
            vecs = vectorizer.transform(words).toarray()  # (num_words, input_dim)
            for v, l in zip(vecs, lab):
                self.samples.append((v, l))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v, l = self.samples[idx]
        return torch.tensor(v, dtype=torch.float32), torch.tensor(l, dtype=torch.float32)

def train(model, dataloader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for ep in range(epochs):
        total = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {ep+1}/{epochs}  loss = {total/len(dataloader):.4f}")

    return model

def infer(model, vectorizer, text: str):
    """
    输入："我 想 订 明天 的 机票"
    输出：[(词, 概率), ...]
    """
    model.eval()
    words = text.split()

    with torch.no_grad():
        x = vectorizer.transform(words).toarray()
        x = torch.tensor(x, dtype=torch.float32)
        scores = model(x).squeeze(1).tolist()

    return list(zip(words, scores))

if __name__ == "__main__":

    # ======== 你在这里填写训练数据即可 ========

    # 示例（你替换成自己的）
    sentences = [
        ["我", "想", "订", "机票"],
        ["帮", "我", "查", "一下", "天气"],
        ["明天", "晚上", "八点", "叫", "我", "起床"],
    ]

    labels = [
        [0, 0, 1, 1],               # “订”“机票”
        [0, 0, 1, 0, 1],            # “查”“天气”
        [1, 0, 0, 1, 0, 1],         # “明天”“叫”“起床”
    ]
    all_words = []
    for s in sentences:
        all_words.extend(s)

    vectorizer = CountVectorizer(tokenizer=lambda x: [x])  # 用词本身，不切字
    vectorizer.fit(all_words)

    # Dataset / DataLoader
    dataset = WordDataset(vectorizer, sentences, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # MLP 模型
    model = KeywordMLP(input_dim=len(vectorizer.vocabulary_))

    # 训练
    model = train(model, dataloader, epochs=20, lr=1e-3)

    # 测试推理
    print("\n=== inference example ===")
    result = infer(model, vectorizer, "我 想 订 明天 的 机票")
    print(result)

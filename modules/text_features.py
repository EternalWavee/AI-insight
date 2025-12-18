# -*- coding: utf-8 -*-
"""
模块 C+D：文本特征工程 + 关键词提取（Pipeline 友好版）
----------------------------------------
目录结构建议：
- assets/audio/   wav 文件
- assets/json/    token JSON + keywords JSON
- 根目录          前端 JSON
"""

import os
import re
import jieba
import jieba.posseg as pseg
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import json

# -------------------------
# 简易 logger
# -------------------------
def get_logger(name):
    class Logger:
        def info(self, msg): print(f"[INFO][{name}] {msg}")
        def warning(self, msg): print(f"[WARN][{name}] {msg}")
    return Logger()

logger = get_logger("text_features")

# -------------------------
# 工具函数：加载停用词
# -------------------------
def load_stopwords(path: str) -> set:
    if not os.path.exists(path):
        logger.warning(f"停用词文件不存在: {path}，将使用空停用词表")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

# -------------------------
# 主类
# -------------------------
class TextFeatureExtractor:
    """
    C+D 模块融合版，pipeline 友好
    """
    def __init__(self, stopwords_path="assets/stopwords.txt"):
        self.stopwords = load_stopwords(stopwords_path)
        self.vectorizer = None
        self.fitted = False
        self.kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

    def tokenize_with_pos(self, text: str):
        tokens = []
        for w, t in pseg.cut(text):
            if w.strip() == "" or w in self.stopwords:
                continue
            tokens.append((w, t))
        return tokens

    def fit_tfidf(self, corpus: list):
        logger.info(f"训练 TF-IDF：语料条数={len(corpus)}")
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x, token_pattern=None, lowercase=False
        )
        self.vectorizer.fit(corpus)
        self.fitted = True
        logger.info("TF-IDF 训练完成")

    def compute_tfidf(self, tokens):
        if not self.fitted:
            raise RuntimeError("TF-IDF 向量器未训练")
        tfidf_vec = self.vectorizer.transform([tokens])
        return {self.vectorizer.get_feature_names_out()[idx]: float(val)
                for idx, val in zip(tfidf_vec.nonzero()[1], tfidf_vec.data)}

    def extract_features(self, text: str, use_sentence_split=True):
        logger.info("开始提取文本特征")
        # 切句
        if use_sentence_split:
            s_raw = re.split(r"([。！？!?])", text)
            merged = [s_raw[i]+s_raw[i+1] if i+1 < len(s_raw) else s_raw[i]
                      for i in range(0, len(s_raw), 2)]
            sentences = [s.strip() for s in merged if s.strip()]
        else:
            sentences = [text]

        all_tokens = []
        for sent_id, sent in enumerate(sentences):
            pos_tokens = self.tokenize_with_pos(sent)
            for idx, (word, pos) in enumerate(pos_tokens):
                all_tokens.append({
                    "token": word,
                    "pos": pos,
                    "sent_id": sent_id,
                    "sent_pos": idx / max(1, len(pos_tokens)-1),
                })
        if not all_tokens:
            logger.warning("文本为空或无有效 token")
            return pd.DataFrame()

        n = len(all_tokens)
        for i in range(n):
            all_tokens[i]["global_pos"] = i / max(1, n-1)

        # 词频
        tokens_only = [x["token"] for x in all_tokens]
        freqs = pd.value_counts(tokens_only).to_dict()
        for x in all_tokens:
            x["freq"] = freqs.get(x["token"], 1)

        # TF-IDF
        tfidf_map = self.compute_tfidf(tokens_only)
        for x in all_tokens:
            x["tfidf"] = tfidf_map.get(x["token"], 0.0)

        # 其他
        for x in all_tokens:
            x["is_digit"] = int(x["token"].isdigit())
            x["len"] = len(x["token"])

        df = pd.DataFrame(all_tokens)
        logger.info(f"文本特征提取完成，共 token={len(df)}")
        return df

    def extract_keywords(self, df_tokens, top_n=10):
        if df_tokens.empty:
            return []
        text_for_kw = " ".join(df_tokens["token"].tolist())
        kws = self.kw_model.extract_keywords(
            text_for_kw,
            keyphrase_ngram_range=(1,1),
            stop_words=list(self.stopwords),
            top_n=top_n
        )
        return [{"keyword": k, "score": float(s)} for k, s in kws]

    # -------------------------
    # pipeline 统一接口
    # -------------------------
    def process_text(self, text: str,
                     tfidf_corpus: list = None,
                     token_json_path: str = "assets/json/text_features.json",
                     keywords_json_path: str = "assets/json/keywords.json"):
        if tfidf_corpus is None:
            tfidf_corpus = [list(jieba.cut(text))]
        self.fit_tfidf(tfidf_corpus)
        df = self.extract_features(text)
        keywords = self.extract_keywords(df, top_n=10)

        # 保存 JSON
        os.makedirs(os.path.dirname(token_json_path), exist_ok=True)
        df.to_json(token_json_path, orient="records", force_ascii=False)
        logger.info(f"token JSON 保存到 {token_json_path}")

        os.makedirs(os.path.dirname(keywords_json_path), exist_ok=True)
        with open(keywords_json_path, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)
        logger.info(f"keywords JSON 保存到 {keywords_json_path}")

        return df, keywords
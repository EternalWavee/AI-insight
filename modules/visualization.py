# -*- coding: utf-8 -*-
"""
模块 F：生成词云 & 输出前端 JSON 框架（Pipeline 友好版）
----------------------------------------
目录结构：
- assets/json/text_features.json
- assets/json/keywords.json
- 根目录 frontend_data.json
- 根目录 wordcloud.png
"""

import os
import json
from wordcloud import WordCloud

CHINESE_FONT_PATH = "MapleMono-NF-CN-Light.ttf"  # 中文字体
WORDCLOUD_OUTPUT = "wordcloud.png"
FRONTEND_JSON_OUTPUT = "frontend_data.json"

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_frontend_json(tokens: list, keywords: list):
    keyword_set = set(kw["keyword"] for kw in keywords)
    sentences = {}
    for tok in tokens:
        sid = str(tok["sent_id"])
        if sid not in sentences:
            sentences[sid] = {
                "sent_id": int(sid),
                "tokens": [],
            }
        sentences[sid]["tokens"].append({
            "token": tok["token"],
            "pos": tok.get("pos",""),
            "tfidf": tok.get("tfidf",0),
            "freq": tok.get("freq",1),
            "is_keyword": tok["token"] in keyword_set
        })
    output_json = {
        "sentences": [sentences[k] for k in sorted(sentences.keys(), key=int)],
        "keywords": keywords
    }
    return output_json

def save_frontend_json(data, path=FRONTEND_JSON_OUTPUT):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 前端 JSON 已保存到 {path}")

def generate_wordcloud(keywords, output_path=WORDCLOUD_OUTPUT):
    wc_dict = {kw["keyword"]: kw["score"] for kw in keywords}
    wc = WordCloud(
        font_path=CHINESE_FONT_PATH,
        width=800,
        height=400,
        background_color="white"
    )
    wc.generate_from_frequencies(wc_dict)
    wc.to_file(output_path)
    print(f"[INFO] 词云图片已保存到 {output_path}")

# -------------------------
# pipeline 统一接口
# -------------------------
def run_visualization(
    token_json_path="assets/json/text_features.json",
    keywords_json_path="assets/json/keywords.json",
    frontend_json_path=FRONTEND_JSON_OUTPUT,
    wordcloud_path=WORDCLOUD_OUTPUT
):
    tokens = load_json(token_json_path)
    keywords = load_json(keywords_json_path)

    frontend_data = build_frontend_json(tokens, keywords)
    save_frontend_json(frontend_data, frontend_json_path)
    generate_wordcloud(keywords, wordcloud_path)

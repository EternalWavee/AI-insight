# -*- coding: utf-8 -*-
"""
Pipeline 三阶段版本
----------------------------------------
支持：
    --stage asr
    --stage keywords
    --stage final
"""

import os
import sys
import json
import argparse
from wordcloud import WordCloud

from modules.asr_whisper import run_asr, ASRError
from modules.text_features import TextFeatureExtractor
import jieba

AUDIO_DIR = "assets/audio"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHINESE_FONT_PATH = "MapleMono-NF-CN-Light.ttf"


# ===============================
# Stage 1: ASR
# ===============================
def stage_asr(wav_path, task_id):

    print(f"[STAGE 1] ASR → {wav_path}")

    try:
        asr_result = run_asr(wav_path)
    except ASRError as e:
        print("[ERROR] ASR failed:", e)
        return

    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_asr.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asr_result, f, ensure_ascii=False, indent=2)

    print(f"[OK] ASR JSON saved: {out_path}")


# ===============================
# Stage 2: keywords
# ===============================
def stage_keywords(wav_path, task_id):

    print(f"[STAGE 2] Extracting keywords")

    # 读取 stage1 输出
    asr_json = os.path.join(OUTPUT_DIR, f"{task_id}_asr.json")
    if not os.path.exists(asr_json):
        print("[ERROR] No ASR JSON for this task.")
        return

    with open(asr_json, "r", encoding="utf-8") as f:
        asr_result = json.load(f)

    text = asr_result["text"]

    extractor = TextFeatureExtractor(stopwords_path="assets/stopwords.txt")

    # TF-IDF 需要语料库
    extractor.fit_tfidf([list(jieba.cut(text))])

    df_tokens = extractor.extract_features(text)

    # 保存 token 特征（可选）
    token_json = os.path.join(OUTPUT_DIR, f"{task_id}_tokens.json")
    df_tokens.to_json(token_json, orient="records", force_ascii=False)

    # 关键词
    keywords = extractor.extract_keywords(df_tokens, top_n=10)

    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_key.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)

    print(f"[OK] Keyword JSON saved: {out_path}")


# ===============================
# Stage 3: final
# ===============================
def stage_final(wav_path, task_id):

    print("[STAGE 3] Building frontend JSON")

    # Load ASR + keywords
    asr_json = os.path.join(OUTPUT_DIR, f"{task_id}_asr.json")
    key_json = os.path.join(OUTPUT_DIR, f"{task_id}_key.json")
    token_json = os.path.join(OUTPUT_DIR, f"{task_id}_tokens.json")

    if not all(os.path.exists(p) for p in [asr_json, key_json, token_json]):
        print("[ERROR] Missing intermediate files")
        return

    with open(asr_json, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    with open(key_json, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    df_tokens = None
    import pandas as pd
    df_tokens = pd.read_json(token_json)

    text = asr_data["text"]

    # ----------- 建 sentence 结构 -----------
    keyword_set = {k["keyword"] for k in keywords}

    sentences = {}
    for tok in df_tokens.to_dict(orient="records"):
        sid = str(tok["sent_id"])
        sentences.setdefault(sid, {
            "sent_id": int(sid),
            "tokens": []
        })
        sentences[sid]["tokens"].append({
            "token": tok["token"],
            "pos": tok.get("pos", ""),
            "tfidf": tok.get("tfidf", 0),
            "freq": tok.get("freq", 1),
            "is_keyword": tok["token"] in keyword_set
        })

    frontend_data = {
        "asr_text": text,
        "sentences": [sentences[k] for k in sorted(sentences.keys(), key=int)],
        "keywords": keywords
    }

    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_frontend.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(frontend_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Final JSON saved: {out_path}")

    # ----------- 词云 -----------
    wc_dict = {kw["keyword"]: kw["score"] for kw in keywords}
    wc = WordCloud(font_path=CHINESE_FONT_PATH,
                   width=800, height=400,
                   background_color="white")

    wc.generate_from_frequencies(wc_dict)
    wc.to_file(os.path.join(OUTPUT_DIR, f"{task_id}_wordcloud.png"))

    print(f"[OK] Wordcloud saved.")

# ===============================
# main (解析 stage)
# ===============================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="音频文件路径")
    parser.add_argument("--stage", required=True,
                        choices=["asr", "keywords", "final"])
    parser.add_argument("--task", required=True,
                        help="task_id（文件名标识）")

    args = parser.parse_args()

    wav_path = args.wav
    task_id = args.task

    if args.stage == "asr":
        stage_asr(wav_path, task_id)
    elif args.stage == "keywords":
        stage_keywords(wav_path, task_id)
    elif args.stage == "final":
        stage_final(wav_path, task_id)

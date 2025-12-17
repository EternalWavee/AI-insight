# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import subprocess
import json
import uuid
import threading

app = Flask(__name__, 
            static_folder='frontend',       # CSS/JS 位于此
            template_folder='frontend')     # index.html 位于此
CORS(app)

UPLOAD_DIR = "assets/audio"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 存所有 task 的阶段状态 ( )
# -------------------------------
TASK_STATE = {} 

def run_pipeline(task_id, wav_path):
    """
    后台执行 Stage2 -> Stage3 ( )
    """
    try:
        # ---- Stage 2: 关键词抽取 ----
        TASK_STATE[task_id]["stage"] = 2
        subprocess.run(
            ["python", "core_pipeline.py", wav_path, "--stage", "keywords", "--task", task_id],
            check=True
        )

        # ---- Stage 3: 生成最终前端数据 ----
        TASK_STATE[task_id]["stage"] = 3
        subprocess.run(
            ["python", "core_pipeline.py", wav_path, "--stage", "final", "--task", task_id],
            check=True
        )

        TASK_STATE[task_id]["done"] = True

    except Exception as e:
        TASK_STATE[task_id]["done"] = True
        TASK_STATE[task_id]["error"] = str(e)

# ----------------------------------------------------
# STEP 0: 渲染主页 (新增路由) 
# ----------------------------------------------------
@app.route("/")
def index():
    # 使用 render_template 渲染 templates/index.html
    # 静态文件路径将通过 Jinja2 的 url_for 自动注入
    return render_template('index.html')


@app.route("/ping")
def ping():
    return "Server running!"


# ----------------------------------------------------
# STEP 1: 上传音频 → whisper → 返回 ASR JSON ( )
# ----------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    task_id = uuid.uuid4().hex
    wav_path = os.path.join(UPLOAD_DIR, f"{task_id}.wav")
    file.save(wav_path)
    print(f"[INFO] Saved audio: {wav_path}")

    # ---------- Stage 1: whisper ----------
    try:
        subprocess.run(
            ["python", "core_pipeline.py", wav_path, "--stage", "asr", "--task", task_id],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ASR failed", "detail": str(e)}), 500

    # 返回 ASR 结果
    asr_path = os.path.join(OUTPUT_DIR, f"{task_id}_asr.json")
    if not os.path.exists(asr_path):
        return jsonify({"error": "ASR output missing"}), 500

    with open(asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)

    TASK_STATE[task_id] = {"stage": 1, "done": False, "error": None}

    # 后台执行 Stage 2-3
    threading.Thread(target=run_pipeline, args=(task_id, wav_path), daemon=True).start()

    return jsonify({
        "status": "ok",
        "task_id": task_id,
        "asr": asr_data 
    })


# ----------------------------------------------------
# STEP 2: 查询处理进度 ( )
# ----------------------------------------------------
@app.route("/status/<task_id>")
def check_status(task_id):
    if task_id not in TASK_STATE:
        return jsonify({"error": "Invalid task_id"}), 404
    return jsonify(TASK_STATE[task_id])


# ----------------------------------------------------
# STEP 3: 获取最终结果 ( )
# ----------------------------------------------------
@app.route("/result/<task_id>")
def result(task_id):
    key_path = os.path.join(OUTPUT_DIR, f"{task_id}_key.json")
    final_path = os.path.join(OUTPUT_DIR, f"{task_id}_frontend.json")
    
    if not os.path.exists(key_path) or not os.path.exists(final_path):
        return jsonify({"error": "Not ready"}), 400

    with open(key_path, "r", encoding="utf-8") as f:
        key_data = json.load(f)
    with open(final_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    wc_url = f"/wordcloud/{task_id}"

    return jsonify({
        "done": True,
        "keywords": key_data,
        "final": final_data,
        "wordcloud": wc_url
    })


# ----------------------------------------------------
# STEP 4: 获取词云图片 ( )
# ----------------------------------------------------
@app.route("/wordcloud/<task_id>")
def get_wordcloud(task_id):
    path = os.path.join(OUTPUT_DIR, f"{task_id}_wordcloud.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "Wordcloud not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
# -*- coding: utf-8 -*-
"""
模块 A：ASR（Whisper）模块
----------------------------------------
功能：
- Whisper 推理
- 段级时间戳
- 近似词/字级时间戳（在 segment 内均匀分配）
- 支持 pipeline：可选 JSON 保存
"""

from __future__ import annotations
import os
import json
import logging
import shutil
from typing import List, Dict, Any

# 限制 CPU 线程（避免 CPU 爆满）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 日志
logger = logging.getLogger("asr_whisper")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ---- ffmpeg 检查 ----
def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "未检测到 ffmpeg，请安装并加入 PATH。\n"
            "Windows 可用 Chocolatey：choco install ffmpeg"
        )

# ---- 依赖检查 ----
try:
    import whisper
except ImportError:
    raise ImportError("请安装 whisper： pip install -U openai-whisper")

# 可选 jieba（中文分词）
_has_jieba = False
try:
    import jieba
    _has_jieba = True
except ImportError:
    _has_jieba = False


# ----------------------- 工具函数 -----------------------
def _is_chinese(text: str) -> bool:
    """粗略判断文本是不是中文为主"""
    if not text:
        return False
    c = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return c / len(text) > 0.3


def _simple_tokenize(text: str) -> List[str]:
    """中文按 jieba → 字，英文按空格"""
    text = text.strip()
    if not text:
        return []

    if _is_chinese(text):
        if _has_jieba:
            return [w for w in jieba.lcut(text) if w.strip()]
        else:
            return [ch for ch in text if ch.strip()]
    else:
        return [w for w in text.split() if w.strip()]


def _uniform_time_splits(start: float, end: float, tokens: List[str]):
    """均匀切时间段"""
    n = max(1, len(tokens))
    total = max(1e-9, end - start)
    per = total / n
    out = []
    for i, t in enumerate(tokens):
        s = start + i * per
        e = start + (i + 1) * per
        out.append({"word": t, "start": float(s), "end": float(e)})
    return out


# ----------------------- 主接口 -----------------------
class ASRError(RuntimeError):
    pass


def run_asr(
    audio_path: str,
    model_name: str = "small",
    language: str | None = None,
    json_path: str | None = None
) -> Dict[str, Any]:
    """
    运行 Whisper ASR 并可选保存 JSON
    返回格式：
    {
      "text": "...",
      "segments": [
        {"start":..., "end":..., "text":"", "words":[{"word":"", "start":.., "end":..}, ...]}
      ]
    }
    """
    # 路径检查
    if not os.path.exists(audio_path):
        raise ASRError(f"音频文件不存在: {audio_path}")

    _check_ffmpeg()

    # 加载模型
    logger.info(f"加载 Whisper 模型 {model_name} (CPU)")
    model = whisper.load_model(model_name, device="cpu")

    # 推理
    logger.info("开始 ASR 推理…")
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, **kwargs)

    text = result.get("text", "").strip()
    segs = result.get("segments", [])

    out_segments = []
    for seg in segs:
        s = float(seg["start"])
        e = float(seg["end"])
        seg_text = seg.get("text", "").strip()
        tokens = _simple_tokenize(seg_text)
        if not tokens:
            words = [{"word": seg_text, "start": s, "end": e}]
        else:
            words = _uniform_time_splits(s, e, tokens)
        out_segments.append({
            "start": s,
            "end": e,
            "text": seg_text,
            "words": words
        })

    out = {"text": text, "segments": out_segments}

    # 保存 JSON
    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"ASR JSON 已保存到 {json_path}")

    return out


# ----------------------- 本地测试 -----------------------
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wav = os.path.join(base, "assets", "audio", "m.wav")
    json_file = os.path.join(base, "assets", "json", "asr_whisper.json")

    try:
        res = run_asr(wav, model_name="small", json_path=json_file)
        print(json.dumps(res, ensure_ascii=False, indent=2)[:2000])
    except Exception as e:
        print("ASR Error:", e)

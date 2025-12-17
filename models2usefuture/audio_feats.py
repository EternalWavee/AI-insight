# modules/audio_feats.py
"""
模块 A — audio_feats.py
功能：
 - 加载音频（支持常见格式）
 - 计算帧级声学特征（RMS、F0(pitch)、ZCR、spectral_centroid）
 - 做能量/静音分段（便于后续与ASR分段/词对齐）
 - 输出帧级和段级特征字典（便于 downstream 使用）

 - 特征工程（Feature Engineering）：提取 RMS、F0、ZCR 等作为输入特征
 - 时间序列分段（Segmentation）：基于能量的静音检测/分段
"""
import os
import math
import traceback
import logging
from typing import Dict, Any, List, Tuple

# 防崩溃设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 标准库和第三方库
import numpy as np
import librosa
import soundfile as sf

# 日志（便于调试）
logger = logging.getLogger("audio_feats")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class AudioFeatureExtractionError(Exception):
    pass


def _time_to_frames(t_seconds: float, sr: int, hop_length: int) -> int:
    return int(round(t_seconds * sr / hop_length))


def _frames_to_time(frame_idx: int, hop_length: int, sr: int) -> float:
    return float(frame_idx * hop_length) / sr


def load_audio(path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    加载音频文件（使用 soundfile + librosa）
    返回 (y, sr)
    """
    try:
        # soundfile 读取原始采样率
        y, file_sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            # 若多声道，转为单声道（取平均）
            y = np.mean(y, axis=1)
        # 如果指定 sr，进行重采样
        if sr is not None and sr != file_sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
            sr = sr
        else:
            sr = file_sr
        # 确保 dtype 是 float32
        y = librosa.util.normalize(y.astype(np.float32))
        return y, sr
    except Exception as e:
        logger.error("load_audio failed: %s", e)
        raise AudioFeatureExtractionError(f"加载音频失败: {e}")


def frame_features(y: np.ndarray,
                   sr: int,
                   frame_ms: float = 25.0,
                   hop_ms: float = 10.0,
                   fmin: float = 50.0,
                   fmax: float = 800.0) -> Dict[str, Any]:
    """
    计算帧级别的声学特征。
    返回 dict:
      {
        'sr': sr,
        'frame_ms': frame_ms,
        'hop_ms': hop_ms,
        'times': np.array([...])   # 每帧的开始时间 (s)
        'rms': np.array([...]),
        'zcr': np.array([...]),
        'centroid': np.array([...]),
        'f0': np.array([...])      # 若无法估计则为 np.nan
      }

    课程映射：特征工程（Feature Engineering）
    """
    try:
        frame_length = int(round(sr * (frame_ms / 1000.0)))
        hop_length = int(round(sr * (hop_ms / 1000.0)))
        if frame_length < 2:
            frame_length = 2
        if hop_length < 1:
            hop_length = 1

        # RMS Energy
        rms = librosa.feature.rms(y=y,
                                  frame_length=frame_length,
                                  hop_length=hop_length,
                                  center=True)[0]  # shape (n_frames,)

        # ZCR (zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(y,
                                                 frame_length=frame_length,
                                                 hop_length=hop_length,
                                                 center=True)[0]

        # Spectral centroid (粗略表征“亮度”)
        centroid = librosa.feature.spectral_centroid(y=y,
                                                     sr=sr,
                                                     n_fft=frame_length,
                                                     hop_length=hop_length,
                                                     center=True)[0]

        # Pitch / F0：使用 librosa.yin（YIN 算法，CPU 友好）
        # yin 返回帧对应的 f0 或 np.nan（unvoiced frames）
        # 注意：yin 的 hop_length 参数是采样点数
        try:
            f0 = librosa.yin(y=y,
                             fmin=fmin,
                             fmax=fmax,
                             sr=sr,
                             frame_length=frame_length,
                             hop_length=hop_length)
            # librosa.yin 可能返回 shape (n_frames,), 保持一致
        except Exception as e:
            logger.warning("librosa.yin failed, falling back to piptrack: %s", e)
            # fallback to piptrack (less robust but works)
            S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length, center=True))
            pitches, magnitudes = librosa.piptrack(S=S, sr=sr, n_fft=frame_length, hop_length=hop_length)
            f0 = np.zeros(pitches.shape[1], dtype=float)
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                f0[i] = pitches[index, i] if magnitudes[index, i] > 0 else np.nan

        # 时间轴（每帧中心或开始时间）
        n_frames = len(rms)
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

        return {
            "sr": sr,
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "frame_length": frame_length,
            "hop_length": hop_length,
            "n_frames": n_frames,
            "times": times,
            "rms": rms,
            "zcr": zcr,
            "centroid": centroid,
            "f0": f0
        }
    except Exception as e:
        logger.error("frame_features failed: %s", e)
        logger.debug(traceback.format_exc())
        raise AudioFeatureExtractionError(f"帧特征提取失败: {e}")


def segments_by_energy(y: np.ndarray,
                       sr: int,
                       top_db: float = 30.0,
                       min_segment_s: float = 0.3) -> List[Tuple[float, float]]:
    """
    基于能量的静音分段（librosa.effects.split）
    返回 segments 列表，每项为 (start_sec, end_sec)

    课程映射：Segmentation（时间序列分段）
    """
    try:
        # returns array of shape (n_intervals, 2) with sample indices
        intervals = librosa.effects.split(y, top_db=top_db)
        segments = []
        for (s, e) in intervals:
            start = float(s / sr)
            end = float(e / sr)
            if (end - start) >= min_segment_s:
                segments.append((start, end))
        # 合并过于接近的段（若 gap 很小）
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                # 如果间隙 < 0.2s，合并
                if seg[0] - prev[1] < 0.2:
                    merged[-1] = (prev[0], seg[1])
                else:
                    merged.append(seg)
        return merged
    except Exception as e:
        logger.error("segments_by_energy failed: %s", e)
        logger.debug(traceback.format_exc())
        raise AudioFeatureExtractionError(f"能量分段失败: {e}")


def segment_level_stats(frame_feats: Dict[str, Any],
                        segments: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    将帧特征汇总到段级，计算每段的平均 RMS、平均 F0 (忽略 nan)、静音占比（帧级）等。
    返回每段的统计信息，便于下游使用（如与 ASR 词时间戳对齐）
    """
    try:
        sr = frame_feats["sr"]
        hop = frame_feats["hop_length"]
        times = frame_feats["times"]
        rms = frame_feats["rms"]
        f0 = frame_feats["f0"]
        zcr = frame_feats["zcr"]
        centroid = frame_feats["centroid"]
        n_frames = frame_feats["n_frames"]

        seg_stats = []
        for (s_sec, e_sec) in segments:
            # 转换为帧 index
            s_frame = max(0, int(np.searchsorted(times, s_sec, side="left")))
            e_frame = min(n_frames, int(np.searchsorted(times, e_sec, side="right")))
            if e_frame <= s_frame:
                # 空段，跳过
                continue
            seg_rms = float(np.mean(rms[s_frame:e_frame]))
            # 平均 F0：忽略 nan
            valid_f0 = f0[s_frame:e_frame]
            if np.all(np.isnan(valid_f0)):
                seg_f0_mean = float("nan")
            else:
                seg_f0_mean = float(np.nanmean(valid_f0))
            seg_zcr = float(np.mean(zcr[s_frame:e_frame]))
            seg_centroid = float(np.mean(centroid[s_frame:e_frame]))
            # 静音帧比例（rms 接近 0）
            silence_frames = np.sum(rms[s_frame:e_frame] < (0.1 * np.max(rms) + 1e-12))
            silence_ratio = float(silence_frames) / (e_frame - s_frame)

            seg_stats.append({
                "start_sec": float(s_sec),
                "end_sec": float(e_sec),
                "duration": float(e_sec - s_sec),
                "start_frame": s_frame,
                "end_frame": e_frame,
                "mean_rms": seg_rms,
                "mean_f0": seg_f0_mean,
                "mean_zcr": seg_zcr,
                "mean_centroid": seg_centroid,
                "silence_ratio": silence_ratio
            })
        return seg_stats
    except Exception as e:
        logger.error("segment_level_stats failed: %s", e)
        logger.debug(traceback.format_exc())
        raise AudioFeatureExtractionError(f"段级统计失败: {e}")


def extract_audio_features(audio_path: str,
                           sr: int = 16000,
                           frame_ms: float = 25.0,
                           hop_ms: float = 10.0,
                           top_db: float = 30.0) -> Dict[str, Any]:
    """
    主函数：加载音频 -> 帧级特征 -> 按能量分段 -> 返回结构化 dict

    返回结构格式示例：
    {
      "audio_path": "...",
      "sr": 16000,
      "duration": 12.34,
      "frame_features": { ... },   # frame_features 的返回
      "segments": [ (s,e), ... ],
      "segment_stats": [ {...}, ... ]
    }

    说明：
     - 该函数为 core_pipeline 提供可直接使用的特征。
     - 后续模块（关键词模型）可以使用 frame_features['times'] 来将词的时间戳对齐到最近帧。
    """
    try:
        logger.info("extract_audio_features start: %s", audio_path)
        y, used_sr = load_audio(audio_path, sr=sr)
        duration = float(len(y) / used_sr)

        # 帧级特征
        ff = frame_features(y, sr=used_sr, frame_ms=frame_ms, hop_ms=hop_ms)

        # 基于能量的分段
        segments = segments_by_energy(y, sr=used_sr, top_db=top_db)

        # 段级统计
        seg_stats = segment_level_stats(ff, segments)

        result = {
            "audio_path": audio_path,
            "sr": used_sr,
            "duration": duration,
            "frame_features": ff,
            "segments": segments,
            "segment_stats": seg_stats
        }
        logger.info("extract_audio_features done: duration=%.2fs, frames=%d, segments=%d",
                    duration, ff["n_frames"], len(segments))
        return result
    except Exception as e:
        logger.error("extract_audio_features failed: %s", e)
        logger.debug(traceback.format_exc())
        raise AudioFeatureExtractionError(f"音频特征提取失败: {e}")


# ---------- Example quick test ----------
if __name__ == "__main__":
    # 仅在直接运行模块时执行简单测试（不影响被导入）
    test_path = "assets/m.wav"
    if not os.path.exists(test_path):
        logger.warning("未找到 sample.wav，跳过本地测试。请把一个示例 wav 放到模块目录并命名为 sample.wav")
    else:
        feats = extract_audio_features(test_path)
        import json
        print(json.dumps({
            "audio_path": feats["audio_path"],
            "sr": feats["sr"],
            "duration": feats["duration"],
            "n_frames": feats["frame_features"]["n_frames"],
            "n_segments": len(feats["segments"])
        }, ensure_ascii=False, indent=2))

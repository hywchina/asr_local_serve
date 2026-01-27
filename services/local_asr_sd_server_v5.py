import os
import uuid
import shutil
import subprocess
import time
from collections import defaultdict

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# =====================================================
# 配置区（核心超参数）
# =====================================================

ASR_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common"
SV_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_sv_zh-cn_16k-common"

TMP_DIR = "/tmp/asr_sd"
os.makedirs(TMP_DIR, exist_ok=True)

DEVICE = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ===== Speaker Tracking 超参数 =====
ORACLE_NUM = 10                 # SD 过切
EMB_SIM_MERGE_TH = 0.88         # 合并阈值
EMB_SIM_CREATE_TH = 0.75        # 新 speaker 阈值
MIN_SEG_DUR = 1.0               # 秒
MAX_SPEAKERS_PER_SESSION = 10   # 问诊场景安全上限


# =====================================================
# Speaker Memory（session 级，全局）
# =====================================================

speaker_memory = defaultdict(dict)
"""
speaker_memory = {
  session_id: {
    global_speaker_id: {
      "embedding": np.ndarray,
      "count": int,
      "last_seen": timestamp
    }
  }
}
"""


# =====================================================
# 工具函数
# =====================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def assign_global_speaker(session_id: str, emb: np.ndarray):
    mem = speaker_memory[session_id]

    best_id = None
    best_sim = 0.0

    for spk_id, info in mem.items():
        sim = cosine_sim(emb, info["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_id = spk_id

    # 1️⃣ 合并到已有 speaker
    if best_sim >= EMB_SIM_MERGE_TH:
        info = mem[best_id]
        info["embedding"] = (
            info["embedding"] * info["count"] + emb
        ) / (info["count"] + 1)
        info["count"] += 1
        info["last_seen"] = time.time()
        return best_id, "merged", best_sim

    # 2️⃣ 创建新 speaker
    if len(mem) < MAX_SPEAKERS_PER_SESSION:
        new_id = f"speaker_{len(mem) + 1}"
        mem[new_id] = {
            "embedding": emb,
            "count": 1,
            "last_seen": time.time()
        }
        return new_id, "new", best_sim

    # 3️⃣ 兜底：强行挂到最相似的
    info = mem[best_id]
    info["embedding"] = (
        info["embedding"] * info["count"] + emb
    ) / (info["count"] + 1)
    info["count"] += 1
    info["last_seen"] = time.time()
    return best_id, "forced_merge", best_sim


def run_ffmpeg(cmd):
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def normalize_audio(input_path: str) -> str:
    out_path = input_path + "_norm.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        out_path
    ]
    run_ffmpeg(cmd)
    return out_path


def cut_audio(input_path, start, end, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    run_ffmpeg(cmd)


# =====================================================
# 模型加载（启动一次）
# =====================================================

print("Loading ASR...")
asr_model = AutoModel(
    model=ASR_MODEL_DIR,
    vad_model=VAD_MODEL_DIR,
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    remote_code="./Fun-ASR/model.py",
    device=DEVICE,
    disable_update=True,
)

print("Loading Speaker Diarization...")
sd_pipeline = pipeline(
    task=Tasks.speaker_diarization,
    model=SD_MODEL_DIR
)

print("Loading Speaker Verification...")
sv_pipeline = pipeline(
    task=Tasks.speaker_verification,
    model=SV_MODEL_DIR
)

print("All models loaded.")


# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="ASR + Speaker Tracking Server",
    version="2.0"
)


@app.post("/asr_sd")
def asr_with_speaker(
    file: UploadFile = File(...),
    session_id: str = Query(..., description="前端传入的会话ID"),
):
    audio_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename)[-1]

    raw_audio = os.path.join(TMP_DIR, audio_id + suffix)
    norm_audio = None

    with open(raw_audio, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        norm_audio = normalize_audio(raw_audio)

        # 1️⃣ Speaker Diarization（过切）
        sd_result = sd_pipeline(
            norm_audio,
            oracle_num=ORACLE_NUM
        )
        sd_result = convert_numpy(sd_result)
        segments = sd_result.get("text", [])

        results = []

        for idx, (start, end, _) in enumerate(segments):
            if end - start < MIN_SEG_DUR:
                continue

            seg_wav = os.path.join(TMP_DIR, f"{audio_id}_{idx}.wav")
            cut_audio(norm_audio, start, end, seg_wav)

            # 2️⃣ ASR
            asr_ret = asr_model.generate(
                input=[seg_wav],
                cache={},
                batch_size=1,
                language="中文",
                itn=True,
                hotwords=[]
            )
            text = asr_ret[0].get("text", "").strip()

            # 3️⃣ Speaker Embedding
            sv_ret = sv_pipeline([seg_wav], output_emb=True)
            sv_ret = convert_numpy(sv_ret)
            emb = np.array(sv_ret["embs"][0], dtype=np.float32)

            # 4️⃣ 全局 speaker 合并
            spk_id, action, sim = assign_global_speaker(session_id, emb)

            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker_id": spk_id,
                "text": text,
                "debug": {
                    "merge_action": action,
                    "similarity": round(sim, 3)
                }
            })

            os.remove(seg_wav)

        return JSONResponse({
            "session_id": session_id,
            "segments": results
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for p in [raw_audio, norm_audio]:
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "local_asr_sd_server_v5:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )

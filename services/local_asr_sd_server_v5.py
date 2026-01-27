import os
import uuid
import shutil
import subprocess
import logging
import torch
import numpy as np
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from funasr import AutoModel
from modelscope.pipelines import pipeline

# =====================================================
# 全局配置
# =====================================================

ASR_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common"

TMP_DIR = "/tmp/asr_sd"
os.makedirs(TMP_DIR, exist_ok=True)

DEVICE = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MIN_SEGMENT_DUR = 0.3  # 秒

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeechEngine")

# =====================================================
# 工具函数
# =====================================================

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


def run_ffmpeg(cmd):
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def normalize_audio(input_path: str) -> str:
    norm_path = input_path + "_norm.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        norm_path
    ]
    run_ffmpeg(cmd)
    return norm_path


def cut_audio(src, start, end, out):
    if end <= start or end - start < MIN_SEGMENT_DUR:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-ar", "16000",
        "-ac", "1",
        out
    ]
    try:
        run_ffmpeg(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


# =====================================================
# 🔥 核心引擎类
# =====================================================

class SpeechEngine:
    """
    统一封装 ASR / SD / 音频处理逻辑
    """

    def __init__(self):
        self._load_models()

    def _load_models(self):
        logger.info("Loading ASR model...")
        self.asr_model = AutoModel(
            model=ASR_MODEL_DIR,
            vad_model=VAD_MODEL_DIR,
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            remote_code="./model.py",
            device=DEVICE,
            disable_update=True,
        )
        logger.info("ASR model loaded.")

        logger.info("Loading Speaker Diarization model...")
        self.sd_pipeline = pipeline(
            task="speaker-diarization",
            model=SD_MODEL_DIR,
            model_revision="v1.0.0"
        )
        logger.info("SD model loaded.")

    # =============================
    # Speaker Diarization
    # =============================
    def diarize(self, wav_path: str) -> List[List[float]]:
        sd_ret = self.sd_pipeline(wav_path, oracle_num=10)
        sd_ret = convert_numpy(sd_ret)
        return sd_ret.get("text", [])

    # =============================
    # ASR 单段
    # =============================
    def asr(self, wav_path: str) -> str:
        ret = self.asr_model.generate(
            input=[wav_path],
            cache={},
            batch_size=1,
            language="中文",
            itn=True
        )
        return ret[0].get("text", "").strip()

    # =============================
    # 主流程
    # =============================
    def process_audio(self, audio_path: str, session_id: str) -> List[Dict]:
        norm_path = normalize_audio(audio_path)
        segments = self.diarize(norm_path)

        results = []
        speaker_map = {}

        for idx, (start, end, spk) in enumerate(segments):
            start, end = float(start), float(end)

            if end - start < MIN_SEGMENT_DUR:
                continue

            if spk not in speaker_map:
                speaker_map[spk] = f"speaker_{len(speaker_map)+1}"

            seg_wav = os.path.join(TMP_DIR, f"{uuid.uuid4()}.wav")
            if not cut_audio(norm_path, start, end, seg_wav):
                continue

            text = self.asr(seg_wav)
            os.remove(seg_wav)

            if not text:
                continue

            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker_id": speaker_map[spk],
                "text": text,
                "session_id": session_id
            })

        os.remove(norm_path)
        return results


# =====================================================
# FastAPI
# =====================================================

engine = SpeechEngine()

app = FastAPI(
    title="ASR + Speaker Diarization Server",
    version="2.0"
)

@app.post("/asr_sd")
def asr_sd(
    file: UploadFile = File(...),
    session_id: str = Query(...)
):
    audio_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename)[-1]
    raw_path = os.path.join(TMP_DIR, audio_id + suffix)

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        segments = engine.process_audio(raw_path, session_id)
        return JSONResponse({
            "session_id": session_id,
            "segments": segments
        })
    except Exception as e:
        logger.exception("Processing failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "local_asr_sd_server_v5:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )

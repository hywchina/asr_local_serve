import os
import uuid
import shutil
import subprocess
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# =====================================================
# 配置区
# =====================================================

ASR_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common"

TMP_DIR = "/tmp/asr_sd"
os.makedirs(TMP_DIR, exist_ok=True)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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


def cut_audio(input_path, start, end, output_path):
    """
    使用 ffmpeg 按时间切音频（秒）
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =====================================================
# 模型加载（只执行一次）
# =====================================================

print("Loading ASR model...")
asr_model = AutoModel(
    model=ASR_MODEL_DIR,
    vad_model=VAD_MODEL_DIR,
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    remote_code="./Fun-ASR/model.py",
    device=DEVICE,
    disable_update=True,
)
print("ASR model loaded.")

print("Loading Speaker Diarization model...")
sd_pipeline = pipeline(
    task=Tasks.speaker_diarization,
    model=SD_MODEL_DIR
)
print("SD model loaded.")

# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="ASR + Speaker Diarization Server",
    version="1.0"
)

@app.post("/asr_sd")
async def asr_with_speaker(file: UploadFile = File(...)):
    audio_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename)[-1]
    audio_path = os.path.join(TMP_DIR, audio_id + suffix)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1️⃣ Speaker Diarization
        sd_result = sd_pipeline(audio_path)
        sd_result = convert_numpy(sd_result)

        segments = sd_result["text"]  # [start, end, speaker_id]

        results = []

        # speaker_id -> 用户1 / 用户2
        speaker_map = {}

        for idx, (start, end, speaker_id) in enumerate(segments):
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"用户{len(speaker_map) + 1}"

            seg_wav = os.path.join(TMP_DIR, f"{audio_id}_{idx}.wav")

            # 2️⃣ 切音频
            cut_audio(audio_path, start, end, seg_wav)

            # 3️⃣ ASR
            asr_ret = asr_model.generate(
                input=[seg_wav],
                cache={},
                batch_size=1,
                language="中文",
                itn=True,
                hotwords=[]
            )

            text = asr_ret[0].get("text", "").strip()

            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": speaker_map[speaker_id],
                "text": text
            })

            os.remove(seg_wav)

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "local_asr_sd_server_v1:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )

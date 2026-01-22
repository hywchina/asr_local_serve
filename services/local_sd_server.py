import os
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# =====================
# 配置
# =====================
MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common"
TMP_DIR = "./tmp_audio"

os.makedirs(TMP_DIR, exist_ok=True)

# =====================
# numpy -> Python 原生类型转换
# =====================
def convert_numpy(obj):
    """
    递归将 numpy 类型转换为 Python 原生类型，避免 FastAPI JSON 序列化失败
    """
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

# =====================
# 加载模型（只加载一次）
# =====================
print("Loading speaker diarization model...")
diarization_pipeline = pipeline(
    task=Tasks.speaker_diarization,
    model=MODEL_DIR
)
print("Model loaded successfully.")

# =====================
# FastAPI
# =====================
app = FastAPI(title="Local Speaker Diarization Server")

@app.post("/diarization")
async def diarization_api(file: UploadFile = File(...)):
    """
    上传音频文件，返回说话人分离结果
    """
    suffix = os.path.splitext(file.filename)[-1]
    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(TMP_DIR, audio_id + suffix)

    # 保存上传的音频
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # 模型推理
    result = diarization_pipeline(audio_path)

    # 关键：转换 numpy 类型
    result = convert_numpy(result)

    # 如不需要保留音频，可开启
    # os.remove(audio_path)

    return {
        "audio_id": audio_id,
        "result": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "local_sd_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )

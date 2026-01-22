import os
import uuid
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from funasr import AutoModel

# =====================================================
# 配置区
# =====================================================

ASR_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"

TMP_AUDIO_DIR = "/tmp/asr_uploads"
os.makedirs(TMP_AUDIO_DIR, exist_ok=True)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =====================================================
# FastAPI 初始化
# =====================================================

app = FastAPI(
    title="Local ASR Server",
    version="1.0",
    description="FunASR local offline service (ASR + VAD)",
)

# =====================================================
# 模型全局加载（⚠️ 只会执行一次）
# =====================================================

asr_model = AutoModel(
    model=ASR_MODEL_DIR,
    vad_model=VAD_MODEL_DIR,               # 本地 VAD
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    remote_code="./Fun-ASR/model.py",
    device=DEVICE,
    disable_update=True,                   # 禁止联网检查
)

# =====================================================
# ASR 接口
# =====================================================

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Form("中文"),
    itn: bool = Form(True),
    hotwords: str = Form(""),
):
    """
    ASR 接口

    - file: 音频文件（wav/mp3）
    - hotwords: 逗号分隔热词，例如 "开放时间,人工客服"
    """

    suffix = os.path.splitext(file.filename)[-1]
    tmp_path = os.path.join(TMP_AUDIO_DIR, f"{uuid.uuid4()}{suffix}")

    try:
        # 1. 保存上传的音频
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 处理 hotwords（⚠️ 永远是 list）
        hotword_list = (
            [w.strip() for w in hotwords.split(",") if w.strip()]
            if hotwords
            else []
        )

        # 3. 调用 FunASR
        result = asr_model.generate(
            input=[tmp_path],
            cache={},
            batch_size=1,
            language=language,
            itn=itn,
            hotwords=hotword_list,   # ✅ 永远不是 None
        )

        text = result[0].get("text", "")

        return JSONResponse(
            content={
                "code": 0,
                "msg": "ok",
                "text": text,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "msg": str(e),
            },
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# =====================================================
# 健康检


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
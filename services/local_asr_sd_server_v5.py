import os
import uuid
import shutil
import subprocess
import logging
import torch
import numpy as np
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')  # 非交互式 backend
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from funasr import AutoModel
from modelscope.pipelines import pipeline

# =====================================================
# 一、全局配置区（模型路径 & 超参数）
# =====================================================

ASR_MODEL_DIR = "/Users/huyanwei/projects/asr_local_serve/models/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/Users/huyanwei/projects/asr_local_serve/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_DIR  = "/Users/huyanwei/projects/asr_local_serve/models/speech_campplus_speaker-diarization_common"
SV_MODEL_DIR  = "/Users/huyanwei/projects/asr_local_serve/models/speech_campplus_sv_zh-cn_16k-common"

ORACLE_NUM = 5
TMP_DIR = "/tmp/asr_sd"
os.makedirs(TMP_DIR, exist_ok=True)

# DEVICE = (
#     "cuda:0" if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available()
#     else "cpu"
# )
DEVICE = "cpu"

# =========================
# 关键超参数
# =========================

MIN_SEGMENT_DUR = 0.3     # SD 切出来的最小语音段（秒）
MIN_EMB_SEG_DUR = 0.8     # 少于该时长的段，不参与 speaker embedding
EMB_SIM_THRESHOLD = 0.5  # embedding 相似度阈值（同一个人的判定）

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeechEngine")

# =====================================================
# 二、工具函数
# =====================================================

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def run_ffmpeg(cmd):
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def normalize_audio(input_path: str) -> str:
    norm_path = input_path + "_norm.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-vn", norm_path]
    run_ffmpeg(cmd)
    return norm_path


def cut_audio(src, start, end, out):
    if end <= start or end - start < MIN_SEGMENT_DUR:
        return False

    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ar", "16000", "-ac", "1", out
    ]
    try:
        run_ffmpeg(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


# =====================================================
# 三、🔥 Speaker Embedding 相似度 & 热力图
# =====================================================

def compute_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    n = len(embeddings)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(embeddings[i], embeddings[j])
    return mat


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    title: str,
    save_path: str
):
    n = sim_matrix.shape[0]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(sim_matrix, cmap="hot", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im)

    plt.xticks(range(n), [f"S{i}" for i in range(n)])
    plt.yticks(range(n), [f"S{i}" for i in range(n)])

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if sim_matrix[i, j] < 0.6 else "black",
                     fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    logger.info(f"Speaker embedding 热力图已保存：{save_path}")


# =====================================================
# 四、🔥 核心类：SpeechEngine
# =====================================================

class SpeechEngine:

    def __init__(self):
        self._load_models()
        self.session_speakers = {}

    def _load_models(self):
        logger.info("加载 ASR 模型...")
        self.asr_model = AutoModel(
            model=ASR_MODEL_DIR,
            vad_model=VAD_MODEL_DIR,
            trust_remote_code=True,
            device=DEVICE,
            dtype="float32",  
            disable_update=True,
            remote_code="./model.py",
        )

        logger.info("加载 SD 模型...")
        self.sd_pipeline = pipeline(
            task="speaker-diarization",
            model=SD_MODEL_DIR,
            model_revision="v1.0.0"
        )

        logger.info("加载 Speaker Embedding 模型...")
        self.sv_pipeline = pipeline(
            task="speaker-verification",
            model=SV_MODEL_DIR
        )

    def match_or_create_speaker(self, session_id: str, emb: np.ndarray):
        bank = self.session_speakers.setdefault(session_id, [])
        for spk in bank:
            sim = cosine_sim(spk["emb"], emb)
            if sim >= EMB_SIM_THRESHOLD:
                spk["emb"] = 0.9 * spk["emb"] + 0.1 * emb
                return spk["id"], sim
        new_id = f"speaker_{len(bank) + 1}"
        bank.append({"id": new_id, "emb": emb})
        return new_id, None

    def process_audio(self, audio_path: str, session_id: str) -> List[Dict]:
        norm_path = normalize_audio(audio_path)

        sd_ret = self.sd_pipeline(norm_path, oracle_num=ORACLE_NUM)
        segments = convert_numpy(sd_ret).get("text", [])
        print(f"debug: segments: {segments}")

        results = []

        # 🔥 用于热力图的 embedding 收集
        emb_list = []
        # 🔁 当前音频内的说话人聚类（不跨会话累计）
        local_bank = []  # 每个元素仅保存聚类中心 emb

        print(f"debug: len(segments): {len(segments)}")
        emb_idx = 0   # 仅统计参与 embedding 的片段数
        seg_idx = 0   # 序号化返回的文本片段
        for start, end, _ in segments:
            start, end = float(start), float(end)
            dur = end - start
            if dur < MIN_SEGMENT_DUR:
                continue

            seg_wav = os.path.join(TMP_DIR, f"{uuid.uuid4()}.wav")
            if not cut_audio(norm_path, start, end, seg_wav):
                continue

            asr_ret = self.asr_model.generate(
                input=[seg_wav], batch_size=1, language="中文", itn=True
            )
            text = asr_ret[0].get("text", "").strip()

            speaker_id = "unknown"
            debug = {}

            if dur >= MIN_EMB_SEG_DUR:
                try:
                    emb = self.sv_pipeline([seg_wav], output_emb=True)["embs"][0]
                    emb = np.array(emb, dtype=np.float32)
                    emb_list.append(emb)

                    print(f"debug:{emb_idx} embedding: {emb}")
                    emb_idx += 1

                    # 在当前音频的局部说话人库中进行匹配/合并
                    matched_id = None
                    matched_sim = None
                    for i, spk in enumerate(local_bank):
                        sim = cosine_sim(spk["emb"], emb)
                        if sim >= EMB_SIM_THRESHOLD:
                            # 轻微更新聚类中心
                            spk["emb"] = 0.9 * spk["emb"] + 0.1 * emb
                            matched_id = f"speaker_{i + 1}"
                            matched_sim = sim
                            break

                    if matched_id is None:
                        local_bank.append({"emb": emb})
                        matched_id = f"speaker_{len(local_bank)}"

                    speaker_id = matched_id
                    debug["merge_similarity"] = round(matched_sim, 3) if matched_sim else None
                except Exception as e:
                    logger.warning(f"Speaker embedding 失败: {e}")

            os.remove(seg_wav)

            if text:
                seg_idx += 1
                results.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "speaker_id": speaker_id,
                    "seg_id": f"S{seg_idx}",
                    "text": text,
                    "debug": debug
                })

        # 🔥 绘制 embedding 相似度热力图
        print(f"debug:emb_list length: {len(emb_list)}")
        heatmap_path = None
        if len(emb_list) >= 2:
            sim_matrix = compute_similarity_matrix(emb_list)
            heatmap_path = os.path.join(
                TMP_DIR, f"speaker_sim_{session_id}.png"
            )
            plot_similarity_heatmap(
                sim_matrix,
                title=f"Session {session_id} Speaker Embedding Similarity",
                save_path=heatmap_path
            )
        if heatmap_path:
            print(f"debug:heatmap_path : {heatmap_path}")
        os.remove(norm_path)
        return results


# =====================================================
# 五、FastAPI
# =====================================================

engine = SpeechEngine()

app = FastAPI(
    title="ASR + SD + Speaker Embedding + Heatmap",
    version="2.2"
)

@app.post("/asr_sd")
def asr_sd(
    file: UploadFile = File(...),
    session_id: str = Query(...)
):
    audio_id = str(uuid.uuid4())
    raw_path = os.path.join(
        TMP_DIR,
        audio_id + os.path.splitext(file.filename)[-1]
    )

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        segments = engine.process_audio(raw_path, session_id)
        return JSONResponse({
            "session_id": session_id,
            "segments": segments
        })
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

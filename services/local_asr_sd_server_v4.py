import os
import uuid
import shutil
import subprocess
import torch
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
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
SPEAKER_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_sv_zh-cn_16k-common"

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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_embeddings(results):
    vectors = []
    speakers = []
    for seg in results:
        emb = seg.get("embedding")
        if emb is None:
            continue
        vectors.append(np.array(emb, dtype=np.float32).reshape(-1))
        speakers.append(seg.get("speaker", ""))

    n = len(vectors)
    if n < 2:
        return {"note": "not_enough_embeddings", "count": n}

    same = []
    diff = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(vectors[i], vectors[j])
            if speakers[i] == speakers[j]:
                same.append(sim)
            else:
                diff.append(sim)

    def stat(values):
        if not values:
            return {"count": 0}
        arr = np.array(values, dtype=np.float32)
        return {
            "count": len(values),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "same_speaker": stat(same),
        "diff_speaker": stat(diff),
    }

    # speaker-level平均相似度矩阵
    spk_to_vecs = {}
    for spk, vec in zip(speakers, vectors):
        spk_to_vecs.setdefault(spk, []).append(vec)
    spk_list = sorted(spk_to_vecs.keys())
    m = len(spk_list)
    matrix = [[None for _ in range(m)] for _ in range(m)]
    for i, si in enumerate(spk_list):
        for j, sj in enumerate(spk_list):
            sims = []
            for va in spk_to_vecs[si]:
                for vb in spk_to_vecs[sj]:
                    if si == sj and va is vb:
                        continue
                    sims.append(cosine_sim(va, vb))
            if sims:
                matrix[i][j] = float(np.mean(sims))
    summary["speaker_matrix"] = {
        "speakers": spk_list,
        "matrix": matrix,
    }

    if diff:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if speakers[i] == speakers[j]:
                    continue
                sim = cosine_sim(vectors[i], vectors[j])
                pairs.append({
                    "sim": sim,
                    "speaker_a": speakers[i],
                    "speaker_b": speakers[j],
                })
        pairs.sort(key=lambda x: x["sim"], reverse=True)
        summary["top_diff_pairs"] = pairs[:5]

    return summary


def save_similarity_heatmap(matrix, speakers, out_path):
    if not MATPLOTLIB_AVAILABLE:
        return None
    m = len(speakers)
    if m == 0:
        return None
    arr = np.full((m, m), np.nan, dtype=np.float32)
    for i in range(m):
        for j in range(m):
            val = matrix[i][j]
            if val is not None:
                arr[i, j] = val

    fig, ax = plt.subplots(figsize=(max(4, m * 0.8), max(3, m * 0.6)))
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(speakers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(speakers, fontsize=8)
    ax.set_xlabel("speaker")
    ax.set_ylabel("speaker")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("cosine", rotation=-90, va="bottom")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_ffmpeg(cmd):
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def normalize_audio(input_path: str) -> str:
    """将任意来源音频统一转为 wav/16k/mono/PCM"""
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


def cut_audio(input_path, start, end, output_path):
    """使用 ffmpeg 按时间切音频（秒）"""
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
    run_ffmpeg(cmd)


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

print("Loading Speaker Embedding model...")
sv_pipeline = pipeline(
    task=Tasks.speaker_verification,
    model=SPEAKER_MODEL_DIR
)
print("Speaker Embedding model loaded.")

# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="ASR + Speaker Diarization Server",
    version="1.1"
)

# ✅ 改为同步函数
@app.post("/asr_sd")
def asr_with_speaker(file: UploadFile = File(...), debug_similarity: bool = False):
    audio_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename)[-1]
    raw_audio_path = os.path.join(TMP_DIR, audio_id + suffix)
    norm_audio_path = None

    with open(raw_audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 0️⃣ 音频统一规范化
        norm_audio_path = normalize_audio(raw_audio_path)

        # 1️⃣ Speaker Diarization
        sd_result = sd_pipeline(norm_audio_path)
        sd_result = convert_numpy(sd_result)

        segments = sd_result.get("text", [])
        results = []
        speaker_map = {}

        for idx, (start, end, speaker_id) in enumerate(segments):
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"用户{len(speaker_map) + 1}"

            seg_wav = os.path.join(TMP_DIR, f"{audio_id}_{idx}.wav")

            # 2️⃣ 切音频
            cut_audio(norm_audio_path, start, end, seg_wav)

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

            # 3.5️⃣ 说话人嵌入
            try:
                sv_ret = sv_pipeline([seg_wav], output_emb=True)
                sv_ret = convert_numpy(sv_ret)
                embedding = sv_ret.get("embs")
                if isinstance(embedding, list) and len(embedding) == 1:
                    embedding = embedding[0]
            except Exception:
                embedding = None

            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": speaker_map[speaker_id],
                "text": text,
                "embedding": embedding
            })

            os.remove(seg_wav)

        if debug_similarity:
            summary = summarize_embeddings(results)
            matrix_info = summary.get("speaker_matrix")
            if matrix_info and MATPLOTLIB_AVAILABLE:
                heatmap_path = os.path.join(TMP_DIR, f"{audio_id}_sim.png")
                saved = save_similarity_heatmap(matrix_info.get("matrix", []), matrix_info.get("speakers", []), heatmap_path)
                summary["heatmap_image"] = saved
            else:
                summary["heatmap_image"] = None if MATPLOTLIB_AVAILABLE else "matplotlib_not_installed"

            return JSONResponse(content={"segments": results, "similarity_debug": summary})

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        for path in [raw_audio_path, norm_audio_path]:
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "local_asr_sd_server_v4:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )

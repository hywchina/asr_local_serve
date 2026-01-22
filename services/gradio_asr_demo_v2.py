import os
import time
import traceback
import requests
import numpy as np
import soundfile as sf
import gradio as gr

API_URL = "http://localhost:8002/asr_sd"

MIN_AUDIO_SEC = 3.0        # ⚠️ 必须 ≥ 后端最小需求
ASR_COOLDOWN = 2.0         # 两次 ASR 至少间隔 2 秒

# =========================
# ASR 调用
# =========================
def call_asr_sd(wav_path):
    with open(wav_path, "rb") as f:
        r = requests.post(
            API_URL,
            files={"file": ("chunk.wav", f, "audio/wav")},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()

# =========================
# UI 渲染
# =========================
def render_dialog(segments, mapping=None, mapped=False):
    html = ""
    for s in segments:
        spk = s["speaker"]
        text = s["text"]

        if not mapped:
            name = spk
            align = "left"
        else:
            info = mapping.get(spk, {"name": spk, "role": "patient"})
            name = info["name"]
            align = "right" if info["role"] in ["doctor", "nurse"] else "left"

        html += f"""
        <div style="
            max-width:70%;
            float:{align};
            clear:both;
            background:#f2f2f2;
            padding:10px;
            margin:6px;
            border-radius:8px;
        ">
        <b>{name}</b><br>{text}
        </div>
        """
    return html or "<i>暂无对话</i>"

# =========================
# 核心：稳定伪实时 ASR
# =========================
def realtime_asr(
    audio_chunk,
    segments,
    audio_buffer,
    buffer_sr,
    last_call_ts,
    asr_busy,
):
    if audio_chunk is None:
        return segments, audio_buffer, buffer_sr, last_call_ts, asr_busy, render_dialog(segments)

    sr, data = audio_chunk
    data = data.astype(np.float32)

    if audio_buffer is None:
        audio_buffer = data
        buffer_sr = sr
    else:
        audio_buffer = np.concatenate([audio_buffer, data])

    duration = len(audio_buffer) / buffer_sr
    now = time.time()

    # ❌ 不满足条件直接返回
    if (
        duration < MIN_AUDIO_SEC
        or asr_busy
        or (now - last_call_ts) < ASR_COOLDOWN
    ):
        return segments, audio_buffer, buffer_sr, last_call_ts, asr_busy, render_dialog(segments)

    # ✅ 进入 ASR
    asr_busy = True
    last_call_ts = now

    tmp = f"/tmp/asr_{int(now*1000)}.wav"
    sf.write(tmp, audio_buffer, buffer_sr)

    try:
        result = call_asr_sd(tmp)
        if isinstance(result, dict):
            result = [result]
        segments.extend(result)
        audio_buffer = None
        buffer_sr = None

    except Exception as e:
        print("⚠️ ASR 失败，等待冷却:", e)
        traceback.print_exc()

    finally:
        asr_busy = False

    return segments, audio_buffer, buffer_sr, last_call_ts, asr_busy, render_dialog(segments)

# =========================
# 身份映射
# =========================
def finish_consult(segments):
    speakers = sorted({s["speaker"] for s in segments})
    mapping = {s: {"name": s, "role": "patient"} for s in speakers}
    return mapping, gr.update(choices=speakers)

def apply_mapping(mapping, spk, role, name):
    mapping[spk] = {"role": role, "name": name}
    return mapping

def rerender(segments, mapping):
    return render_dialog(segments, mapping, mapped=True)

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    segments = gr.State([])
    audio_buffer = gr.State(None)
    buffer_sr = gr.State(None)
    last_call_ts = gr.State(0.0)
    asr_busy = gr.State(False)
    mapping = gr.State({})

    gr.Markdown("## 🎙 实时问诊（稳定版）")

    audio = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="点击麦克风开始问诊"
    )

    dialog = gr.HTML("<i>暂无对话</i>")

    audio.stream(
        realtime_asr,
        inputs=[
            audio,
            segments,
            audio_buffer,
            buffer_sr,
            last_call_ts,
            asr_busy,
        ],
        outputs=[
            segments,
            audio_buffer,
            buffer_sr,
            last_call_ts,
            asr_busy,
            dialog,
        ],
    )

    gr.Markdown("### 👤 说话人身份（结束问诊后）")

    finish = gr.Button("结束问诊")
    spk_dd = gr.Dropdown(label="说话人")
    role_dd = gr.Dropdown(["doctor", "nurse", "patient", "family"])
    name_tb = gr.Textbox(label="姓名")
    apply = gr.Button("应用")

    finish.click(finish_consult, segments, [mapping, spk_dd])
    apply.click(apply_mapping, [mapping, spk_dd, role_dd, name_tb], mapping)
    apply.click(rerender, [segments, mapping], dialog)

demo.launch()

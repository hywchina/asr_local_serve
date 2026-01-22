import os
import time
from typing import Dict, List
import gradio as gr
import requests

# =========================
# 后端 ASR + 说话人识别 API
# =========================
API_URL = os.getenv("ASR_SD_API", "http://localhost:8002/asr_sd")
TIMEOUT = 300


def call_asr_sd(audio_path: str) -> List[Dict]:
    with open(audio_path, "rb") as f:
        r = requests.post(
            API_URL,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=TIMEOUT,
        )
    r.raise_for_status()
    return r.json()


# =========================
# 状态常量
# =========================
IDLE = "idle"
RUNNING = "running"
PAUSED = "paused"
FINISHED = "finished"
MAPPED = "mapped"


# =========================
# 工具函数
# =========================
def extract_speaker_ids(segments):
    return sorted({seg["speaker"] for seg in segments})


def render_transcript(segments, speaker_map=None):
    """
    - 映射前：只显示 speakerId
    - 映射后：根据身份左右区分显示
    """
    lines = []

    for seg in segments:
        speaker_id = seg["speaker"]
        text = seg["text"]

        if speaker_map and speaker_id in speaker_map:
            role = speaker_map[speaker_id]["role"]
            name = speaker_map[speaker_id]["name"]

            if role in ["doctor", "nurse"]:
                prefix = f"➡️ **{name}（医护）**"
            else:
                prefix = f"⬅️ **{name}**"
        else:
            prefix = f"**{speaker_id}**"

        lines.append(f"{prefix}：{text}")

    return "\n\n".join(lines) if lines else "暂无对话"


# =========================
# 问诊流程（generator，直接驱动 UI）
# =========================
def consultation_pipeline(
    audio_path,
    consult_state,
    segments_state,
    speaker_map_state,
):
    if not audio_path:
        yield consult_state, segments_state, "请先录音", "暂无对话"
        return

    # 调用后端（一次性获取分段）
    all_segments = call_asr_sd(audio_path)

    consult_state = RUNNING
    segments_state = []

    # 前端“假实时”流式转录（等价 HTML setInterval）
    for seg in all_segments:
        if consult_state == PAUSED:
            while consult_state == PAUSED:
                time.sleep(0.2)

        if consult_state == FINISHED:
            break

        segments_state.append(seg)

        yield (
            consult_state,
            segments_state,
            "问诊中…",
            render_transcript(segments_state, speaker_map_state),
        )

        time.sleep(0.8)

    consult_state = FINISHED
    yield (
        consult_state,
        segments_state,
        "问诊结束，请进行说话人身份设置",
        render_transcript(segments_state, speaker_map_state),
    )


# =========================
# 控制按钮逻辑
# =========================
def pause_or_resume(consult_state):
    if consult_state == RUNNING:
        return PAUSED, "已暂停"
    if consult_state == PAUSED:
        return RUNNING, "继续问诊"
    return consult_state, "当前状态不可暂停/继续"


def stop_consult(consult_state):
    if consult_state in [RUNNING, PAUSED]:
        return FINISHED, "问诊已手动结束"
    return consult_state, "当前状态不可结束"


# =========================
# 身份映射
# =========================
def init_mapping(segments):
    mapping = {}
    for i, spk in enumerate(extract_speaker_ids(segments)):
        mapping[spk] = {
            "role": "patient" if i == 0 else "doctor",
            "name": spk,
        }
    return mapping


def apply_mapping(speaker, role, name, mapping):
    mapping[speaker] = {
        "role": role,
        "name": name or speaker,
    }
    return mapping


# =========================
# 报告生成（前端占位）
# =========================
def generate_report(segments, mapping, consult_state):
    if consult_state != MAPPED:
        return (
            "❌ 请先完成说话人身份设置",
            "",
            "",
        )

    patient_texts = []
    doctor_texts = []

    for seg in segments:
        role = mapping[seg["speaker"]]["role"]
        if role in ["patient", "family"]:
            patient_texts.append(seg["text"])
        elif role in ["doctor", "nurse"]:
            doctor_texts.append(seg["text"])

    return (
        " ".join(patient_texts),
        " ".join(doctor_texts),
        "（诊断建议：后续由大模型生成）",
    )


# =========================
# Gradio UI
# =========================
def build_demo():
    with gr.Blocks(title="智能医生问诊系统（Gradio）") as demo:

        gr.Markdown("## 🏥 智能医生问诊系统")

        status_md = gr.Markdown("准备就绪")

        consult_state = gr.State(IDLE)
        segments_state = gr.State([])
        speaker_map_state = gr.State({})

        audio = gr.Audio(sources=["microphone"], type="filepath")

        with gr.Row():
            start_btn = gr.Button("开始问诊", variant="primary")
            pause_btn = gr.Button("暂停 / 继续")
            stop_btn = gr.Button("结束问诊")

        transcript_md = gr.Markdown("暂无对话")

        # ========= 身份映射 =========
        gr.Markdown("### 👤 说话人身份设置（问诊结束后）")

        speaker_dd = gr.Dropdown(label="说话人")
        role_dd = gr.Dropdown(
            ["patient", "family", "doctor", "nurse"],
            label="身份",
        )
        name_tb = gr.Textbox(label="显示名称")
        apply_btn = gr.Button("应用身份设置")

        # ========= 报告 =========
        gr.Markdown("### 📄 结构化问诊报告")

        patient_report = gr.Textbox(label="病人自述")
        doctor_report = gr.Textbox(label="医生问诊摘要")
        diagnosis_report = gr.Textbox(label="初步诊断建议")

        generate_btn = gr.Button("生成结构化报告")

        # ======================
        # 事件绑定
        # ======================

        # 开始问诊（直接绑定 generator）
        start_btn.click(
            consultation_pipeline,
            inputs=[audio, consult_state, segments_state, speaker_map_state],
            outputs=[consult_state, segments_state, status_md, transcript_md],
        )

        # 暂停 / 继续
        pause_btn.click(
            pause_or_resume,
            inputs=consult_state,
            outputs=[consult_state, status_md],
        )

        # 结束问诊
        stop_btn.click(
            stop_consult,
            inputs=consult_state,
            outputs=[consult_state, status_md],
        )

        # 问诊结束 → 初始化身份映射
        consult_state.change(
            lambda st, segs: init_mapping(segs) if st == FINISHED else gr.update(),
            inputs=[consult_state, segments_state],
            outputs=speaker_map_state,
        )

        # 更新 speaker 下拉框
        speaker_map_state.change(
            lambda m: gr.update(choices=list(m.keys())),
            speaker_map_state,
            speaker_dd,
        )

        # 选择 speaker → 填充表单
        speaker_dd.change(
            lambda s, m: (m[s]["role"], m[s]["name"]),
            inputs=[speaker_dd, speaker_map_state],
            outputs=[role_dd, name_tb],
        )

        # 应用身份映射（可多次）
        apply_btn.click(
            apply_mapping,
            inputs=[speaker_dd, role_dd, name_tb, speaker_map_state],
            outputs=speaker_map_state,
        )

        # 映射完成 → 状态变为 MAPPED + 重排对话
        speaker_map_state.change(
            lambda m, segs: (
                MAPPED,
                render_transcript(segs, m),
            ),
            inputs=[speaker_map_state, segments_state],
            outputs=[consult_state, transcript_md],
        )

        # 生成报告
        generate_btn.click(
            generate_report,
            inputs=[segments_state, speaker_map_state, consult_state],
            outputs=[patient_report, doctor_report, diagnosis_report],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)

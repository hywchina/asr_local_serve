import os
from typing import Any, Dict, List, Tuple
import gradio as gr
import requests

API_URL = os.getenv("ASR_SD_API", "http://localhost:8002/asr_sd")
TIMEOUT = int(os.getenv("ASR_SD_TIMEOUT", "300"))

def call_asr_sd(audio_path: str) -> List[Dict[str, Any]]:
    """Send audio file to local ASR + SD service and return parsed JSON."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=TIMEOUT,
        )
    if not resp.ok:
        snippet = resp.text[:300]
        raise RuntimeError(f"HTTP {resp.status_code}: {snippet}")
    return resp.json()


def format_segments(data: List[Dict[str, Any]]) -> Tuple[str, List[List[Any]]]:
    """Build markdown summary and table rows from service response."""
    rows: List[List[Any]] = []
    lines: List[str] = []

    for seg in sorted(data, key=lambda s: s.get("start", 0)):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "说话人")
        text = seg.get("text", "")

        rows.append([speaker, f"{start:.2f}-{end:.2f}", text])
        lines.append(f"**{speaker}** ({start:.2f}-{end:.2f}s): {text}")

    transcript = "\n".join(lines) if lines else "未识别到文本。"
    return transcript, rows


def run_pipeline(audio_path: str):
    if not audio_path:
        yield "准备就绪，点击录音开始。", "", [], None
        return

    yield "识别中，请稍候...", "", [], None

    try:
        data = call_asr_sd(audio_path)
        transcript, rows = format_segments(data if isinstance(data, list) else [])
        yield "识别完成。", transcript, rows, data
    except Exception as exc:
        message = f"请求失败，请确认后端已启动。错误: {exc}"
        yield message, "", [], {"error": str(exc)}


def build_demo() -> gr.Blocks:
    custom_css = """
    body { background: radial-gradient(circle at 20% 20%, #2f2f2f, #0e0e0e 60%); color: #f5f5f5; }
    .gradio-container { max-width: 1100px; margin: auto; }
    .status-card { padding: 10px 15px; background: #1c1c1c; border: 1px solid #333; border-radius: 12px; margin-bottom: 10px; }
    .transcript-card { padding: 12px; background: #222; border-radius: 12px; border: 1px solid #444; }
    .table-card { background: #222; border-radius: 12px; border: 1px solid #444; padding: 5px; }
    """

    with gr.Blocks(title="本地语音识别 Demo") as demo:
        gr.Markdown("## 本地语音识别 + 说话人分离\n使用麦克风录音，点击下方按钮进行识别。")

        # 状态显示
        status_md = gr.Markdown("准备就绪，点击录音开始。", elem_classes=["status-card"])

        with gr.Row():
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                format="wav",
                label="点击录音完成后点击下方识别",
            )
            with gr.Column():
                submit_btn = gr.Button("开始识别", variant="primary")
                clear_btn = gr.Button("清空", variant="secondary")

        # 转写文本和分段表格
        with gr.Row():
            transcript_md = gr.Markdown("", label="转写文本", elem_classes=["transcript-card"])
            segments_table = gr.Dataframe(
                headers=["说话人", "时间段(s)", "文本"],
                datatype=["str", "str", "str"],
                label="分段结果",
                wrap=True,
                elem_classes=["table-card"]
            )

        raw_json = gr.JSON(label="原始返回")

        # 按钮绑定
        submit_btn.click(
            fn=run_pipeline,
            inputs=audio,
            outputs=[status_md, transcript_md, segments_table, raw_json],
        )

        def reset_ui():
            return "准备就绪，点击录音开始。", "", [], None

        clear_btn.click(
            fn=reset_ui,
            outputs=[status_md, transcript_md, segments_table, raw_json],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css="""
        body { background: radial-gradient(circle at 20% 20%, #2f2f2f, #0e0e0e 60%); color: #f5f5f5; }
        .gradio-container { max-width: 1100px; margin: auto; }
        .status-card { padding: 10px 15px; background: #1c1c1c; border: 1px solid #333; border-radius: 12px; margin-bottom: 10px; }
        .transcript-card { padding: 12px; background: #222; border-radius: 12px; border: 1px solid #444; }
        .table-card { background: #222; border-radius: 12px; border: 1px solid #444; padding: 5px; }
        """,
    )

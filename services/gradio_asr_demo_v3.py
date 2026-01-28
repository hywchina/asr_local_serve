"""
智能医生问诊AI系统 - Gradio前端
兼容 Gradio 6.3.0
"""

import gradio as gr
import requests
import numpy as np
import uuid
import time
import io
from typing import List, Dict, Optional
from datetime import datetime
import soundfile as sf

# =====================================================
# 配置参数
# =====================================================

BACKEND_URL = "http://localhost:8002"
CHUNK_DURATION = 12  # 每12秒发送一次音频到后端（提高有效语音时长，避免SD过短报错）
SAMPLE_RATE = 16000
MAX_BUFFER_DURATION = 30  # 最长缓存时长，避免无限增长

# =====================================================
# 全局状态
# =====================================================

class SessionState:
    def __init__(self):
        self.session_id = None
        self.is_recording = False
        self.is_paused = False
        self.audio_buffer = []
        self.all_segments = []
        self.speaker_mapping = {}
        self.last_send_time = None
        self.chunk_counter = 0
        
    def start_new_session(self):
        self.session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        self.is_recording = True
        self.is_paused = False
        self.audio_buffer = []
        self.all_segments = []
        self.speaker_mapping = {}
        self.last_send_time = time.time()
        self.chunk_counter = 0
        
        try:
            requests.post(f"{BACKEND_URL}/reset_session", 
                         params={"session_id": self.session_id}, timeout=5)
        except:
            pass

state = SessionState()

# =====================================================
# 工具函数
# =====================================================

def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()


def send_audio_to_backend(audio_data: np.ndarray, session_id: str) -> List[Dict]:
    try:
        wav_bytes = numpy_to_wav_bytes(audio_data, SAMPLE_RATE)
        files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
        params = {'session_id': session_id}
        
        print(f"Sending {len(audio_data)/SAMPLE_RATE:.1f}s audio to backend...")
        response = requests.post(f"{BACKEND_URL}/asr_sd", files=files, params=params, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            segments = result.get('segments', [])
            print(f"Received {len(segments)} segments")
            return segments
        else:
            print(f"Backend error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Failed to send audio: {e}")
        return []


def merge_adjacent_segments(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        if seg['speaker_id'] == current['speaker_id']:
            current['end'] = seg['end']
            current['text'] += " " + seg['text']
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged


def get_unique_speakers() -> List[str]:
    speakers = set()
    for seg in state.all_segments:
        speaker_id = seg.get('speaker_id')
        if speaker_id and speaker_id != 'unknown':
            speakers.add(speaker_id)
    return sorted(list(speakers))


def format_segment_text(seg: Dict, use_mapping: bool = False) -> str:
    """格式化单个片段为文本"""
    speaker_id = seg.get('speaker_id', 'unknown')
    text = seg.get('text', '')
    start = seg.get('start', 0)
    end = seg.get('end', 0)
    
    if use_mapping and speaker_id in state.speaker_mapping:
        display_name = state.speaker_mapping[speaker_id]
    else:
        display_name = speaker_id
    
    return f"[{start:.1f}s-{end:.1f}s] {display_name}: {text}"


def build_conversation_text(use_mapping: bool = False) -> str:
    """构建对话文本显示"""
    if not state.all_segments:
        return "暂无对话记录"
    
    segments = state.all_segments
    if not state.is_recording:
        segments = merge_adjacent_segments(segments)
    
    lines = []
    for seg in segments:
        lines.append(format_segment_text(seg, use_mapping))
    
    return "\n\n".join(lines)


# =====================================================
# 按钮事件处理
# =====================================================

def start_consultation():
    state.start_new_session()
    return (
        gr.update(interactive=False),  # start_btn
        gr.update(interactive=True),   # pause_btn
        gr.update(interactive=True),   # end_btn
        "",  # conversation_display
        gr.update(visible=False),  # speaker_settings
        gr.update(interactive=False),  # report_btn
        f"✅ 问诊已开始 | 会话ID: {state.session_id}"
    )


def pause_consultation():
    if state.is_paused:
        state.is_paused = False
        return gr.update(value="⏸️ 暂停"), "▶️ 问诊已恢复"
    else:
        state.is_paused = True
        return gr.update(value="▶️ 继续"), "⏸️ 问诊已暂停"


def end_consultation():
    state.is_recording = False
    state.is_paused = False
    
    conversation_text = build_conversation_text(use_mapping=False)
    speakers = get_unique_speakers()
    
    return (
        gr.update(interactive=True),   # start_btn
        gr.update(interactive=False),  # pause_btn
        gr.update(interactive=False),  # end_btn
        conversation_text,  # conversation_display
        gr.update(visible=True),  # speaker_settings
        gr.update(interactive=False),  # report_btn
        f"🏁 问诊已结束 | 检测到 {len(speakers)} 位说话人"
    )


def apply_speaker_mapping(speaker_inputs):
    """应用说话人身份映射"""
    speakers = get_unique_speakers()
    
    # 解析输入的映射
    state.speaker_mapping = {}
    for line in speaker_inputs.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                spk_id = parts[0].strip()
                spk_name = parts[1].strip()
                if spk_id in speakers and spk_name:
                    state.speaker_mapping[spk_id] = spk_name
    
    conversation_text = build_conversation_text(use_mapping=True)
    
    return (
        conversation_text,
        gr.update(interactive=True),
        f"✅ 已应用 {len(state.speaker_mapping)} 位说话人的身份映射"
    )


def reset_speaker_mapping():
    state.speaker_mapping = {}
    conversation_text = build_conversation_text(use_mapping=False)
    return (
        conversation_text,
        gr.update(interactive=False),
        "🔄 身份映射已重置"
    )


def generate_speaker_mapping_template():
    """生成说话人映射模板"""
    speakers = get_unique_speakers()
    if not speakers:
        return "暂无说话人"
    
    template_lines = []
    for speaker in speakers:
        template_lines.append(f"{speaker}: ")
    
    return "\n".join(template_lines)


def generate_report():
    report = f"""# 问诊报告

**会话ID:** {state.session_id}
**问诊时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**说话人数量:** {len(get_unique_speakers())}

## 说话人身份
"""
    if state.speaker_mapping:
        for k, v in state.speaker_mapping.items():
            report += f"- {k}: {v}\n"
    else:
        report += "未设置身份映射\n"
    
    report += "\n## 对话记录\n\n"
    
    segments = merge_adjacent_segments(state.all_segments)
    for seg in segments:
        speaker_id = seg['speaker_id']
        display_name = state.speaker_mapping.get(speaker_id, speaker_id)
        report += f"**{display_name}** [{seg['start']:.1f}s - {seg['end']:.1f}s]\n\n{seg['text']}\n\n"
    
    return report


# =====================================================
# 音频流处理
# =====================================================

def process_audio_stream(audio):
    if not state.is_recording or state.is_paused:
        return build_conversation_text(use_mapping=False), "等待录音..."
    
    if audio is None:
        return build_conversation_text(use_mapping=False), "未检测到音频"
    
    sample_rate, audio_data = audio
    
    # 转单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 重采样
    if sample_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sample_rate
        new_length = int(len(audio_data) * ratio)
        audio_data = np.interp(
            np.linspace(0, len(audio_data) - 1, new_length),
            np.arange(len(audio_data)),
            audio_data
        )
    
    # 归一化
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0
    
    state.audio_buffer.append(audio_data)
    
    buffer_duration = sum(len(chunk) for chunk in state.audio_buffer) / SAMPLE_RATE
    status_msg = f"录音中... 缓冲区: {buffer_duration:.1f}s"
    
    if buffer_duration >= CHUNK_DURATION:
        full_audio = np.concatenate(state.audio_buffer)
        
        status_msg = f"正在发送音频块 #{state.chunk_counter + 1}..."
        segments = send_audio_to_backend(full_audio, state.session_id)
        
        if segments:
            state.all_segments.extend(segments)
            state.chunk_counter += 1
            status_msg = f"✅ 已接收块 #{state.chunk_counter} | 检测到 {len(segments)} 个片段"
            overlap_samples = int(0.5 * SAMPLE_RATE)
            if len(full_audio) > overlap_samples:
                state.audio_buffer = [full_audio[-overlap_samples:]]
            else:
                state.audio_buffer = []
        else:
            # 未返回片段时，保留缓存以累积更长语音（避免丢音）
            status_msg = "⚠️ 语音有效时长不足，继续累积中..."
            # 限制最大缓存长度，防止内存增长
            max_samples = int(MAX_BUFFER_DURATION * SAMPLE_RATE)
            if len(full_audio) > max_samples:
                state.audio_buffer = [full_audio[-max_samples:]]
        
        state.last_send_time = time.time()
    
    return build_conversation_text(use_mapping=False), status_msg


# =====================================================
# Gradio 界面
# =====================================================

with gr.Blocks(title="智能医生问诊AI系统") as demo:
    gr.Markdown("# 🏥 智能医生问诊AI系统")
    gr.Markdown("基于语音识别和说话人分离的实时问诊记录系统")
    
    gr.Markdown("""
    <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <strong>⚠️ 使用提示：</strong>
        <ul style="margin: 5px 0;">
            <li>请使用 <code>http://localhost:7860</code> 或 <code>http://127.0.0.1:7860</code> 访问以启用麦克风</li>
            <li>首次使用时浏览器会请求麦克风权限，请点击"允许"</li>
            <li>确保后端服务已启动（端口8002）</li>
        </ul>
    </div>
    """)
    
    with gr.Row():
        # 左侧
        with gr.Column(scale=2):
            status_display = gr.Textbox(
                label="系统状态",
                value="准备就绪，点击「开始问诊」开始录音",
                interactive=False
            )
            
            with gr.Row():
                start_btn = gr.Button("🎙️ 开始问诊", variant="primary")
                pause_btn = gr.Button("⏸️ 暂停", interactive=False)
                end_btn = gr.Button("🏁 结束问诊", interactive=False)
            
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="录音输入"
            )
            
            gr.Markdown("### 💬 实时对话转录")
            conversation_display = gr.Textbox(
                label="对话记录",
                lines=20,
                interactive=False,
                placeholder="对话内容将在这里显示..."
            )
        
        # 右侧
        with gr.Column(scale=1):
            gr.Markdown("### 👥 说话人身份设置")
            
            with gr.Column(visible=False) as speaker_settings:
                gr.Markdown("请为每位说话人设置身份（格式：speaker_1: 张医生）")
                
                mapping_template_btn = gr.Button("生成模板", size="sm")
                
                speaker_mapping_input = gr.Textbox(
                    label="说话人身份映射",
                    lines=8,
                    placeholder="speaker_1: 张医生\nspeaker_2: 患者李某\nspeaker_3: 家属",
                    interactive=True
                )
                
                with gr.Row():
                    apply_btn = gr.Button("✅ 应用设置", variant="primary")
                    reset_btn = gr.Button("🔄 重置")
            
            gr.Markdown("### 📋 结构化报告")
            report_btn = gr.Button("生成报告", interactive=False)
            report_output = gr.Markdown(value="")
    
    # 事件绑定
    start_btn.click(
        fn=start_consultation,
        outputs=[start_btn, pause_btn, end_btn, conversation_display, 
                speaker_settings, report_btn, status_display]
    )
    
    pause_btn.click(
        fn=pause_consultation,
        outputs=[pause_btn, status_display]
    )
    
    end_btn.click(
        fn=end_consultation,
        outputs=[start_btn, pause_btn, end_btn, conversation_display,
                speaker_settings, report_btn, status_display]
    )
    
    audio_input.stream(
        fn=process_audio_stream,
        inputs=[audio_input],
        outputs=[conversation_display, status_display],
        stream_every=1.0
    )
    
    mapping_template_btn.click(
        fn=generate_speaker_mapping_template,
        outputs=[speaker_mapping_input]
    )
    
    apply_btn.click(
        fn=apply_speaker_mapping,
        inputs=[speaker_mapping_input],
        outputs=[conversation_display, report_btn, status_display]
    )
    
    reset_btn.click(
        fn=reset_speaker_mapping,
        outputs=[conversation_display, report_btn, status_display]
    )
    
    report_btn.click(
        fn=generate_report,
        outputs=[report_output]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("智能医生问诊AI系统 - Gradio 6.3.0")
    print("=" * 60)
    print("📍 本地访问: http://localhost:7860")
    print("⚠️  麦克风需要HTTPS或localhost访问")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
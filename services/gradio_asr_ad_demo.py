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
from typing import List, Dict
from datetime import datetime
import soundfile as sf

# =====================================================
# 配置参数
# =====================================================

BACKEND_URL = "http://localhost:8002"
CHUNK_DURATION = 30  # 每30秒发送一次音频到后端（提高有效语音时长，避免SD过短报错）
SAMPLE_RATE = 16000
MAX_BUFFER_DURATION = 30  # 最长缓存时长，避免无限增长
MIN_VOICED_DURATION = 3.0  # 最小有效语音时长（秒），低于该值不发送后端
VAD_FRAME_MS = 30          # 能量检测帧长（毫秒）
VAD_ENERGY_THRESHOLD = 0.01  # 能量阈值（经验值）
VAD_WINDOW_DURATION = 8.0   # 只在最近窗口估算有效语音时长（秒）
SILENCE_RESET_SECONDS = 6.0  # 连续静音超过该值则重置缓存
MIN_SEND_DURATION = 6.0      # 允许更短的发送窗口（秒）
MIN_SEND_INTERVAL = 2.0      # 发送最小间隔（秒）
MIN_VOICED_RATIO = 0.2       # 最近窗口内语音占比阈值

# Debug
DEBUG_ENABLED = True
DEBUG_HISTORY_LIMIT = 200
DEBUG_SHOW_LINES = 8

# 说话人身份配置
MAX_SPEAKER_SLOTS = 10
ROLE_OPTIONS = [
    "医生", "护士", "技师", "药师", "其他(医院)",
    "患者", "陪诊", "家属", "其他(患者)"
]
HOSPITAL_ROLES = {"医生", "护士", "技师", "药师", "其他(医院)"}

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
        self.last_voiced_time = None
        self.last_display_count = 0
        self.debug_events = []
        self.last_segment_sig = None
        
    def start_new_session(self):
        self.session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        self.is_recording = True
        self.is_paused = False
        self.audio_buffer = []
        self.all_segments = []
        self.speaker_mapping = {}
        self.last_send_time = time.time()
        self.chunk_counter = 0
        self.last_voiced_time = time.time()
        self.last_display_count = 0
        self.debug_events = []
        self.last_segment_sig = None
        
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


def debug_log(message: str):
    if not DEBUG_ENABLED:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    state.debug_events.append(f"[{ts}] {message}")
    if len(state.debug_events) > DEBUG_HISTORY_LIMIT:
        state.debug_events = state.debug_events[-DEBUG_HISTORY_LIMIT:]


def debug_summary() -> str:
    if not DEBUG_ENABLED or not state.debug_events:
        return ""
    return "\n".join(state.debug_events[-DEBUG_SHOW_LINES:])


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




def build_conversation_messages(use_mapping: bool = False, use_layout: bool = False):
    """构建 Chatbot 消息列表"""
    if not state.all_segments:
        return []

    segments = merge_adjacent_segments(state.all_segments)

    messages = []
    for seg in segments:
        speaker_id = seg.get("speaker_id", "unknown")
        text = seg.get("text", "")
        if use_mapping and speaker_id in state.speaker_mapping:
            mapping = state.speaker_mapping[speaker_id]
            display_name = mapping.get("name") or mapping.get("role") or speaker_id
        else:
            display_name = speaker_id
        content = f"{display_name}: {text}".strip()

        if use_layout and use_mapping:
            role = state.speaker_mapping.get(speaker_id, {}).get("role")
            if role in HOSPITAL_ROLES:
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})
        else:
            # 未设置身份前统一左侧展示
            messages.append({"role": "assistant", "content": content})

    return messages


def estimate_voiced_duration(audio_data: np.ndarray) -> float:
    """基于能量的简易VAD，估算有效语音时长（秒）"""
    if audio_data.size == 0:
        return 0.0

    frame_len = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
    if frame_len <= 0:
        return 0.0

    total_frames = int(np.ceil(len(audio_data) / frame_len))
    voiced_frames = 0

    for i in range(total_frames):
        start = i * frame_len
        end = min(start + frame_len, len(audio_data))
        frame = audio_data[start:end]
        if frame.size == 0:
            continue
        energy = float(np.sqrt(np.mean(frame ** 2)))
        if energy >= VAD_ENERGY_THRESHOLD:
            voiced_frames += 1

    voiced_seconds = (voiced_frames * frame_len) / SAMPLE_RATE
    return voiced_seconds


# =====================================================
# 按钮事件处理
# =====================================================

def start_consultation():
    state.start_new_session()
    speaker_updates = []
    for _ in range(MAX_SPEAKER_SLOTS):
        speaker_updates.append(gr.update(visible=False))
    speaker_id_updates = [gr.update(value="") for _ in range(MAX_SPEAKER_SLOTS)]
    speaker_role_updates = [gr.update(value=None, interactive=False) for _ in range(MAX_SPEAKER_SLOTS)]
    speaker_name_updates = [gr.update(value="", interactive=False) for _ in range(MAX_SPEAKER_SLOTS)]

    return (
        gr.update(interactive=False),  # start_btn
        gr.update(interactive=True),   # pause_btn
        gr.update(interactive=True),   # end_btn
        [],  # conversation_display
        gr.update(visible=False),  # speaker_settings
        gr.update(interactive=False),  # report_btn
        f"✅ 问诊已开始 | 会话ID: {state.session_id}",
        *speaker_updates,
        *speaker_id_updates,
        *speaker_role_updates,
        *speaker_name_updates
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

    conversation_messages = build_conversation_messages(use_mapping=False, use_layout=False)
    state.last_display_count = len(conversation_messages)
    speakers = get_unique_speakers()

    speaker_updates = []
    speaker_id_updates = []
    speaker_role_updates = []
    speaker_name_updates = []

    for i in range(MAX_SPEAKER_SLOTS):
        if i < len(speakers):
            speaker_updates.append(gr.update(visible=True))
            speaker_id_updates.append(gr.update(value=speakers[i]))
            speaker_role_updates.append(gr.update(value=None, interactive=True))
            speaker_name_updates.append(gr.update(value="", interactive=True))
        else:
            speaker_updates.append(gr.update(visible=False))
            speaker_id_updates.append(gr.update(value=""))
            speaker_role_updates.append(gr.update(value=None, interactive=False))
            speaker_name_updates.append(gr.update(value="", interactive=False))
    
    return (
        gr.update(interactive=True),   # start_btn
        gr.update(interactive=False),  # pause_btn
        gr.update(interactive=False),  # end_btn
        conversation_messages,  # conversation_display
        gr.update(visible=True),  # speaker_settings
        gr.update(interactive=False),  # report_btn
        f"🏁 问诊已结束 | 检测到 {len(speakers)} 位说话人",
        *speaker_updates,
        *speaker_id_updates,
        *speaker_role_updates,
        *speaker_name_updates
    )


def apply_speaker_mapping(*inputs):
    """应用说话人身份映射（下拉选择）"""
    speakers = set(get_unique_speakers())
    total = len(inputs)
    third = total // 3
    speaker_ids = inputs[:third]
    speaker_roles = inputs[third: third * 2]
    speaker_names = inputs[third * 2:]

    state.speaker_mapping = {}
    for spk_id, role, name in zip(speaker_ids, speaker_roles, speaker_names):
        if spk_id in speakers and role:
            state.speaker_mapping[spk_id] = {
                "role": role,
                "name": (name or "").strip()
            }

    conversation_messages = build_conversation_messages(use_mapping=True, use_layout=True)
    state.last_display_count = len(conversation_messages)

    return (
        conversation_messages,
        gr.update(interactive=True),
        f"✅ 已应用 {len(state.speaker_mapping)} 位说话人的身份映射"
    )


def reset_speaker_mapping():
    state.speaker_mapping = {}
    conversation_messages = build_conversation_messages(use_mapping=False, use_layout=False)
    state.last_display_count = len(conversation_messages)
    return (
        conversation_messages,
        gr.update(interactive=False),
        "🔄 身份映射已重置"
    )


def generate_report():
    report = f"""# 问诊报告

**会话ID:** {state.session_id}
**问诊时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**说话人数量:** {len(get_unique_speakers())}

## 说话人身份
"""
    if state.speaker_mapping:
        for k, v in state.speaker_mapping.items():
            role = v.get("role")
            name = v.get("name")
            label = f"{role} {name}".strip()
            report += f"- {k}: {label}\n"
    else:
        report += "未设置身份映射\n"
    
    report += "\n## 对话记录\n\n"
    
    segments = merge_adjacent_segments(state.all_segments)
    for seg in segments:
        speaker_id = seg['speaker_id']
        if speaker_id in state.speaker_mapping:
            mapping = state.speaker_mapping[speaker_id]
            display_name = mapping.get("name") or mapping.get("role") or speaker_id
        else:
            display_name = speaker_id
        report += f"**{display_name}** [{seg['start']:.1f}s - {seg['end']:.1f}s]\n\n{seg['text']}\n\n"
    
    return report


# =====================================================
# 音频流处理
# =====================================================

def process_audio_stream(audio):
    prev_sig = state.last_segment_sig
    if not state.is_recording or state.is_paused:
        return gr.update(), "等待录音..."
    
    if audio is None:
        return gr.update(), "未检测到音频"
    
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

    full_audio = np.concatenate(state.audio_buffer)
    # 仅检查最近窗口的有效语音，避免长时间静音拖累
    window_samples = int(VAD_WINDOW_DURATION * SAMPLE_RATE)
    if len(full_audio) > window_samples:
        recent_audio = full_audio[-window_samples:]
    else:
        recent_audio = full_audio

    recent_duration = len(recent_audio) / SAMPLE_RATE if len(recent_audio) else 0.0
    voiced_duration = estimate_voiced_duration(recent_audio)
    voiced_ratio = (voiced_duration / recent_duration) if recent_duration > 0 else 0.0
    debug_log(
        f"buffer={buffer_duration:.1f}s recent={recent_duration:.1f}s voiced={voiced_duration:.1f}s ratio={voiced_ratio:.2f}"
    )

    if voiced_ratio >= MIN_VOICED_RATIO:
        state.last_voiced_time = time.time()
    elif state.last_voiced_time and (time.time() - state.last_voiced_time) >= SILENCE_RESET_SECONDS:
        # 连续静音过久，清空缓存，避免历史静音拖累识别
        state.audio_buffer = []
        state.last_send_time = None
        state.last_voiced_time = None
        debug_log("silence_reset buffer cleared")

    should_send = False
    if (
        voiced_duration >= MIN_VOICED_DURATION
        and voiced_ratio >= MIN_VOICED_RATIO
        and buffer_duration >= MIN_SEND_DURATION
    ):
        if buffer_duration >= CHUNK_DURATION:
            should_send = True
        elif state.last_send_time is None or (time.time() - state.last_send_time) >= MIN_SEND_INTERVAL:
            should_send = True
    debug_log(f"should_send={should_send}")

    if should_send:
        if voiced_duration < MIN_VOICED_DURATION or voiced_ratio < MIN_VOICED_RATIO:
            status_msg = f"⚠️ 有效语音 {voiced_duration:.1f}s 不足，继续累积中..."
            max_samples = int(MAX_BUFFER_DURATION * SAMPLE_RATE)
            if len(full_audio) > max_samples:
                state.audio_buffer = [full_audio[-max_samples:]]
            debug_log("skip_send: insufficient voiced")
            return gr.update(), f"{status_msg}\n{debug_summary()}" if DEBUG_ENABLED else status_msg

        status_msg = f"正在发送音频块 #{state.chunk_counter + 1}..."
        audio_to_send = full_audio if buffer_duration >= CHUNK_DURATION else recent_audio
        debug_log(f"send_audio duration={len(audio_to_send)/SAMPLE_RATE:.1f}s")
        segments = send_audio_to_backend(audio_to_send, state.session_id)
        debug_log(f"segments_received={len(segments)}")
        
        if segments:
            state.all_segments.extend(segments)
            state.chunk_counter += 1
            status_msg = f"✅ 已接收块 #{state.chunk_counter} | 检测到 {len(segments)} 个片段"
            overlap_samples = int(0.5 * SAMPLE_RATE)
            if len(audio_to_send) > overlap_samples:
                state.audio_buffer = [audio_to_send[-overlap_samples:]]
            else:
                state.audio_buffer = []
            debug_log(f"total_segments={len(state.all_segments)}")
        else:
            # 未返回片段时，保留缓存以累积更长语音（避免丢音）
            status_msg = "⚠️ 语音有效时长不足，继续累积中..."
            # 限制最大缓存长度，防止内存增长
            max_samples = int(MAX_BUFFER_DURATION * SAMPLE_RATE)
            if len(full_audio) > max_samples:
                state.audio_buffer = [full_audio[-max_samples:]]
            if voiced_ratio < MIN_VOICED_RATIO and state.last_voiced_time is None:
                state.audio_buffer = []
                state.last_send_time = None
                debug_log("no_segments: silent_reset")
            else:
                debug_log("no_segments: keep_buffer")
        
        state.last_send_time = time.time()
    
    messages = build_conversation_messages(use_mapping=False, use_layout=False)
    if state.all_segments:
        last_seg = state.all_segments[-1]
        current_sig = (
            last_seg.get("speaker_id"),
            last_seg.get("start"),
            last_seg.get("end"),
            last_seg.get("text")
        )
    else:
        current_sig = None

    if len(messages) == state.last_display_count and current_sig == prev_sig:
        return gr.update(), status_msg

    state.last_display_count = len(messages)
    state.last_segment_sig = current_sig
    if DEBUG_ENABLED:
        status_msg = f"{status_msg}\n{debug_summary()}"
    return messages, status_msg


# =====================================================
# Gradio 界面
# =====================================================

CHAT_CSS = """
#conversation_chatbot {
    height: 520px;
}
#conversation_chatbot .wrap {
    height: 520px;
    overflow-y: auto;
}
"""

with gr.Blocks(title="智能医生问诊AI系统") as demo:
    gr.HTML(
        """
        <script>
        const scrollChatbotToBottom = () => {
            const wrap = document.querySelector('#conversation_chatbot .wrap');
            if (wrap) {
                wrap.scrollTop = wrap.scrollHeight;
            }
        };
        const observer = new MutationObserver(scrollChatbotToBottom);
        const setupObserver = () => {
            const target = document.querySelector('#conversation_chatbot');
            if (target) {
                observer.observe(target, { childList: true, subtree: true });
            }
        };
        window.addEventListener('load', () => {
            setupObserver();
            setInterval(scrollChatbotToBottom, 1000);
        });
        </script>
        """
    )
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
            conversation_display = gr.Chatbot(
                label="对话记录",
                height=520,
                elem_id="conversation_chatbot"
            )
        
        # 右侧
        with gr.Column(scale=1):
            gr.Markdown("### 👥 说话人身份设置")
            
            with gr.Column(visible=False) as speaker_settings:
                gr.Markdown("问诊结束后，请为检测到的说话人分配身份。选择为医院人员的消息将显示在右侧，其它身份显示在左侧。")

                speaker_rows = []
                speaker_id_boxes = []
                speaker_role_dropdowns = []
                speaker_name_inputs = []

                for i in range(MAX_SPEAKER_SLOTS):
                    with gr.Row(visible=False) as row:
                        spk_id = gr.Textbox(label="说话人ID", interactive=False)
                        spk_role = gr.Dropdown(
                            choices=ROLE_OPTIONS,
                            label="身份选择",
                            value=None,
                            interactive=True
                        )
                        spk_name = gr.Textbox(
                            label="姓名",
                            placeholder="例如：王医生 / 患者李某",
                            interactive=True
                        )
                    speaker_rows.append(row)
                    speaker_id_boxes.append(spk_id)
                    speaker_role_dropdowns.append(spk_role)
                    speaker_name_inputs.append(spk_name)

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
            speaker_settings, report_btn, status_display,
            *speaker_rows, *speaker_id_boxes, *speaker_role_dropdowns, *speaker_name_inputs]
    )
    
    pause_btn.click(
        fn=pause_consultation,
        outputs=[pause_btn, status_display]
    )
    
    end_btn.click(
        fn=end_consultation,
        outputs=[start_btn, pause_btn, end_btn, conversation_display,
            speaker_settings, report_btn, status_display,
            *speaker_rows, *speaker_id_boxes, *speaker_role_dropdowns, *speaker_name_inputs]
    )
    
    audio_input.stream(
        fn=process_audio_stream,
        inputs=[audio_input],
        outputs=[conversation_display, status_display],
        stream_every=1.0
    )
    
    apply_btn.click(
        fn=apply_speaker_mapping,
        inputs=[*speaker_id_boxes, *speaker_role_dropdowns, *speaker_name_inputs],
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
        share=False,
        css=CHAT_CSS
    )
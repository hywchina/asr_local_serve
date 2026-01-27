import gradio as gr
import json
import time
import uuid
import requests
import threading
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import io
import wave

# ==================== 配置 ====================
BACKEND_API_URL = "http://localhost:8002/asr_sd"
CHUNK_DURATION = 30  # 每次录音分段时长(秒) — 拉长以保持说话人连续性
SAMPLE_RATE = 16000  # 采样率
MIN_AUDIO_LENGTH = SAMPLE_RATE * CHUNK_DURATION  # 最小音频长度(采样点数)
ROLE_OPTIONS = ["患者", "家属/陪诊", "医生", "护士", "其他"]
MAX_SPEAKERS = 4  # 预留最多4个说话人下拉行

# ==================== 全局状态管理 ====================
class ConsultationState:
    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.transcripts = []  # 存储所有转录内容
        self.speaker_mapping = {}  # 说话人ID到身份的映射
        self.unique_speakers = set()  # 检测到的说话人集合
        self.mapping_done = False  # 是否完成身份映射
        self.start_time = None
        self.session_id = None  # 后端会话ID
        self.speaker_label_map = {}  # backend speaker_id -> 用户X
        self.audio_buffer = []  # 累积的音频数据(numpy array)
        self.recording_thread = None
        self.processed_chunks = 0  # 已处理的音频块数量
        self.total_audio_samples = 0  # 累积的音频采样点总数
        self.transcript_version = 0  # 用于减少无意义UI刷新
        self.last_rendered_version = -1
        self.last_ui_render_time = 0.0
        self.cached_transcript_html = ""
        self.buffer_lock = threading.Lock()  # 保护音频缓冲区的锁
        self.is_processing = False  # 标记是否正在处理音频块
        
    def reset(self):
        self.__init__()

state = ConsultationState()

# ==================== 后端API调用 ====================
def call_backend_api(audio_data: bytes) -> List[Dict]:
    """
    调用后端ASR+SD API
    audio_data: WAV格式的音频字节流
    返回: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    """
    try:
        files = {"file": ("chunk.wav", audio_data, "audio/wav")}
        # 后端v5要求session_id,用UUID保持同一问诊会话
        if not state.session_id:
            state.session_id = uuid.uuid4().hex

        response = requests.post(
            BACKEND_API_URL,
            params={"session_id": state.session_id},
            files=files,
            timeout=30
        )

        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return []

        data = response.json()
        segments = data.get("segments", []) if isinstance(data, dict) else data

        normalized = []
        for seg in segments:
            speaker_id = seg.get("speaker_id") or seg.get("speaker") or "unknown"
            display = state.speaker_label_map.setdefault(
                speaker_id,
                f"用户{len(state.speaker_label_map) + 1}"
            )

            normalized.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": display,
                "text": seg.get("text", "").strip()
            })

        return normalized
    except Exception as e:
        print(f"Backend API call failed: {str(e)}")
        return []

def convert_audio_to_wav(audio_array, sample_rate=16000):
    """
    将音频数组转换为WAV格式的字节流
    """
    buffer = io.BytesIO()
    
    # 确保音频数据是int16格式
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    buffer.seek(0)
    return buffer.read()


def has_enough_voice(audio_array: np.ndarray, sample_rate: int, min_voice_seconds: float = 3.0, energy_threshold: float = 0.01) -> Tuple[bool, float]:
    """简单能量检测: 有效语音时长不足时跳过调用后端,避免无效请求"""
    if audio_array.size == 0:
        return False, 0.0

    audio = audio_array.astype(np.float32)
    # 如果是int16,缩放到[-1,1]
    if audio.max(initial=0) > 1.5:
        audio = audio / 32768.0

    frame = max(int(0.02 * sample_rate), 1)  # 20ms帧
    if len(audio) < frame:
        return False, 0.0

    # 滑动均方根能量
    window = np.ones(frame) / frame
    rms = np.sqrt(np.convolve(audio ** 2, window, mode="valid"))
    voiced = rms > energy_threshold
    voiced_duration = voiced.sum() * (frame / sample_rate)

    return voiced_duration >= min_voice_seconds, float(voiced_duration)

# ==================== 核心功能函数 ====================

def format_time(seconds: float) -> str:
    """格式化时间为 MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def format_transcript_html(transcripts: List[Dict], mapping_done: bool, speaker_mapping: Dict) -> str:
    """格式化对话转录为HTML"""
    if not transcripts:
        return '<div style="color: #999; text-align: center; padding: 20px;">暂无对话内容</div>'
    
    html = '<div style="display: flex; flex-direction: column; gap: 15px; max-height: 500px; overflow-y: auto; padding: 10px;">'
    
    for item in transcripts:
        speaker_label = item['speaker']  # "用户1", "用户2" 等
        text = item['text']
        time_range = f"{format_time(item['start'])}-{format_time(item['end'])}"
        
        # 确定说话人显示名称
        if mapping_done and speaker_label in speaker_mapping:
            speaker_name = speaker_mapping[speaker_label]['name']
            role = speaker_mapping[speaker_label]['role']
            is_hospital = role in ['doctor', 'nurse']
        else:
            speaker_name = speaker_label
            is_hospital = False
        
        # 确定消息位置和样式
        if mapping_done:
            if is_hospital:
                # 医院方 - 右侧,蓝色
                align = "flex-end"
                bg_color = "#e3f2fd"
                name_color = "#1e90ff"
                border_radius = "12px 12px 4px 12px"
            else:
                # 患者方 - 左侧,绿色
                align = "flex-start"
                bg_color = "#e8f5e9"
                name_color = "#4caf50"
                border_radius = "12px 12px 12px 4px"
        else:
            # 未映射 - 左侧,灰色
            align = "flex-start"
            bg_color = "#f0f0f0"
            name_color = "#666"
            border_radius = "12px 12px 12px 4px"
        
        html += f'''
        <div style="display: flex; justify-content: {align};">
            <div style="
                max-width: 80%;
                padding: 12px 16px;
                background-color: {bg_color};
                border-radius: {border_radius};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 12px; font-weight: 600; color: {name_color}; margin-bottom: 5px;">
                    {speaker_name}
                </div>
                <div style="font-size: 14px; line-height: 1.6; color: #333;">
                    {text}
                </div>
                <div style="font-size: 11px; color: #999; text-align: right; margin-top: 5px;">
                    {time_range}
                </div>
            </div>
        </div>
        '''
    
    html += '</div>'
    return html

def merge_consecutive_speakers(transcripts: List[Dict]) -> List[Dict]:
    """合并相邻同一说话人的内容"""
    if not transcripts:
        return []
    
    merged = []
    current = transcripts[0].copy()
    
    for i in range(1, len(transcripts)):
        if transcripts[i]['speaker'] == current['speaker']:
            # 合并文本和时间
            current['text'] += ' ' + transcripts[i]['text']
            current['end'] = transcripts[i]['end']
        else:
            merged.append(current)
            current = transcripts[i].copy()
    
    merged.append(current)
    return merged

def create_speaker_mapping_ui(speaker_labels: List[str]) -> List[List]:
    """创建说话人身份映射UI的数据"""
    if not speaker_labels:
        return []
    
    rows = []
    for idx, speaker_label in enumerate(sorted(speaker_labels)):
        # 默认猜测:用户1可能是患者,用户2可能是医生
        if speaker_label == "用户1":
            default_role = "患者"
            default_name = "患者1"
        elif speaker_label == "用户2":
            default_role = "医生"
            default_name = "医生"
        else:
            default_role = "其他"
            default_name = speaker_label
        
        rows.append([speaker_label, default_role, default_name])
    
    return rows


def build_mapping_updates(mapping_data: List[List]) -> Tuple:
    """将映射数据行转换为控件update; 输出顺序: 全部label, 全部role, 全部name"""
    if not mapping_data:
        mapping_data = []

    labels = ["" for _ in range(MAX_SPEAKERS)]
    roles = ["患者" for _ in range(MAX_SPEAKERS)]
    names = ["" for _ in range(MAX_SPEAKERS)]

    for idx, row in enumerate(mapping_data[:MAX_SPEAKERS]):
        if len(row) >= 1:
            labels[idx] = row[0]
        if len(row) >= 2 and row[1] in ROLE_OPTIONS:
            roles[idx] = row[1]
        if len(row) >= 3:
            names[idx] = row[2]

    label_updates = []
    role_updates = []
    name_updates = []
    for i in range(MAX_SPEAKERS):
        visible = bool(labels[i])
        label_updates.append(gr.update(value=labels[i], visible=visible))
        role_updates.append(gr.update(value=roles[i], choices=ROLE_OPTIONS, visible=visible))
        name_updates.append(gr.update(value=names[i], visible=visible))
    return tuple(label_updates + role_updates + name_updates)


def build_empty_mapping_updates() -> Tuple:
    label_updates = []
    role_updates = []
    name_updates = []
    for _ in range(MAX_SPEAKERS):
        label_updates.append(gr.update(value="", visible=False))
        role_updates.append(gr.update(value="患者", choices=ROLE_OPTIONS, visible=False))
        name_updates.append(gr.update(value="", visible=False))
    return tuple(label_updates + role_updates + name_updates)

# ==================== 音频处理和实时转录 ====================

def process_audio_chunk(audio_chunk, sample_rate):
    """
    处理单个音频块:
    1. 转换为WAV格式
    2. 调用后端API
    3. 更新转录结果
    """
    try:
        # 转换音频格式
        wav_data = convert_audio_to_wav(audio_chunk, sample_rate)
        
        # 调用后端API
        results = call_backend_api(wav_data)
        
        if results:
            # 计算时间偏移(基于已处理的音频块)
            time_offset = state.processed_chunks * CHUNK_DURATION
            
            # 调整时间戳并添加到转录列表
            for item in results:
                adjusted_item = {
                    'speaker': item['speaker'],
                    'text': item['text'],
                    'start': item['start'] + time_offset,
                    'end': item['end'] + time_offset
                }
                state.transcripts.append(adjusted_item)
                state.unique_speakers.add(item['speaker'])
            
            state.processed_chunks += 1
            state.transcript_version += 1  # 标记有新内容,触发UI刷新
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing audio chunk: {str(e)}")
        return False

# ==================== 事件处理函数 ====================

def start_consultation():
    """开始问诊"""
    state.reset()
    state.session_id = uuid.uuid4().hex  # 为本次问诊生成session_id
    state.is_recording = True
    state.start_time = datetime.now()
    state.cached_transcript_html = format_transcript_html([], False, {})
    state.last_rendered_version = state.transcript_version
    state.last_ui_render_time = time.time()
    
    return (
        gr.update(interactive=False),  # 开始按钮
        gr.update(interactive=True),   # 暂停按钮
        gr.update(interactive=True),   # 结束按钮
        gr.update(value="<div style='color: #4cd964;'>● 问诊中...</div>"),  # 状态指示
        format_transcript_html([], False, {}),  # 清空对话
        gr.update(value="提示:问诊结束后,请为检测到的说话人分配身份。设置为\"医生\"或\"护士\"身份的消息将显示在右侧,其他身份消息显示在左侧。"),
        gr.update(interactive=False),  # 应用设置按钮
        gr.update(interactive=False),  # 重置设置按钮
        gr.update(interactive=False),  # 生成报告按钮
        gr.update(value=""),  # 清空报告
        f"0 条对话",  # 对话计数
        "等待录音...",  # 音频状态
        gr.update(visible=False),  # 音频组件占位
        None  # 占位: 映射数据
    )

def pause_consultation():
    """暂停/继续问诊"""
    state.is_paused = not state.is_paused
    
    if state.is_paused:
        status_text = "<div style='color: #ff9500;'>⏸ 已暂停</div>"
        pause_btn_text = "▶️ 继续"
    else:
        status_text = "<div style='color: #4cd964;'>● 问诊中...</div>"
        pause_btn_text = "⏸ 暂停"
    
    return gr.update(value=status_text), gr.update(value=pause_btn_text)

def stop_consultation():
    """结束问诊"""
    state.is_recording = False
    
    # 等待当前处理完成
    while state.is_processing:
        time.sleep(0.1)
    
    # 🔒 处理缓冲区中的剩余音频（加锁保护）
    with state.buffer_lock:
        if state.audio_buffer and state.total_audio_samples > 0:
            combined_audio = np.concatenate(state.audio_buffer)
            state.audio_buffer = []
            state.total_audio_samples = 0
        else:
            combined_audio = None
    
    if combined_audio is not None and len(combined_audio) > SAMPLE_RATE:  # 至少1秒
        voice_ok, voiced_secs = has_enough_voice(combined_audio, SAMPLE_RATE)
        if voice_ok:
            process_audio_chunk(combined_audio, SAMPLE_RATE)
        else:
            print(f"Skip final chunk: voiced {voiced_secs:.2f}s < min")
    
    # 等待录音线程结束
    if state.recording_thread and state.recording_thread.is_alive():
        state.recording_thread.join(timeout=2)
    
    # 合并相邻同一说话人的对话
    state.transcripts = merge_consecutive_speakers(state.transcripts)
    
    # 统计说话人
    state.unique_speakers = set(t['speaker'] for t in state.transcripts)
    
    # 创建映射表格数据
    mapping_data = create_speaker_mapping_ui(list(state.unique_speakers))
    
    speaker_count = len(state.unique_speakers)
    
    return (
        gr.update(interactive=True),   # 开始按钮
        gr.update(interactive=False, value="⏸ 暂停"),  # 暂停按钮
        gr.update(interactive=False),  # 结束按钮
        gr.update(value="<div style='color: #666;'>✓ 问诊已结束</div>"),  # 状态
        format_transcript_html(state.transcripts, False, {}),  # 刷新对话显示
        mapping_data,
        gr.update(value=f"问诊结束,共检测到 {speaker_count} 个说话人。请设置说话人身份后点击\"应用设置\"。"),
        gr.update(interactive=True),   # 启用应用设置
        gr.update(interactive=True),   # 启用重置设置
        gr.update(interactive=False),  # 生成报告仍禁用
        f"{len(state.transcripts)} 条对话",
        "录音已结束"
    )

def on_audio_stream(audio_data):
    """
    实时音频流处理回调 - 累积音频到足够长度后再处理
    audio_data: tuple (sample_rate, audio_array)
    """
    now = time.time()
    
    # 🐛 调试：检查音频数据
    if audio_data is not None:
        print(f"[DEBUG] 收到音频数据: type={type(audio_data)}, is_recording={state.is_recording}, is_paused={state.is_paused}")
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            print(f"[DEBUG] sample_rate={audio_data[0]}, audio_shape={audio_data[1].shape if hasattr(audio_data[1], 'shape') else 'no shape'}")

    if not state.is_recording or state.is_paused:
        with state.buffer_lock:
            current_samples = state.total_audio_samples
        transcript_html = state.cached_transcript_html or format_transcript_html(
            state.transcripts, state.mapping_done, state.speaker_mapping
        )
        return (
            gr.update(value=transcript_html),
            f"{len(state.transcripts)} 条对话",
            f"累积音频: {current_samples / SAMPLE_RATE:.1f}秒"
        )
    
    if audio_data is None:
        print("[DEBUG] audio_data is None")
        with state.buffer_lock:
            current_samples = state.total_audio_samples
        transcript_html = state.cached_transcript_html or format_transcript_html(
            state.transcripts, state.mapping_done, state.speaker_mapping
        )
        return (
            gr.update(value=transcript_html),
            f"{len(state.transcripts)} 条对话",
            f"累积音频: {current_samples / SAMPLE_RATE:.1f}秒"
        )
    
    sample_rate, audio_array = audio_data
    
    # 转换为单声道(如果是立体声)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # 🔒 始终累积音频到缓冲区（加锁保护）
    with state.buffer_lock:
        state.audio_buffer.append(audio_array)
        state.total_audio_samples += len(audio_array)
        current_duration = state.total_audio_samples / SAMPLE_RATE
        should_process = (state.total_audio_samples >= MIN_AUDIO_LENGTH 
                         and not state.is_processing)
    
    # 🐛 调试输出
    if current_duration > 0 and int(current_duration) % 5 == 0 and current_duration < int(current_duration) + 0.5:
        print(f"[DEBUG] 累积音频: {current_duration:.1f}秒, 需要: {MIN_AUDIO_LENGTH/SAMPLE_RATE:.1f}秒, should_process={should_process}, is_processing={state.is_processing}")
    
    status_msg = f"累积音频: {current_duration:.1f}秒"
    
    # 检查是否累积到足够长度且当前没有在处理
    transcript_html = state.cached_transcript_html or format_transcript_html(
        state.transcripts, state.mapping_done, state.speaker_mapping
    )
    html_update = gr.update(value=transcript_html)

    if should_process:
        # 标记为处理中
        state.is_processing = True
        
        # 🔒 获取当前缓冲区并清空（加锁保护）
        with state.buffer_lock:
            combined_audio = np.concatenate(state.audio_buffer)
            state.audio_buffer = []  # 清空缓冲区，新音频会继续累积到新buffer
            state.total_audio_samples = 0
        
        # 🚀 在锁外进行耗时操作（不阻塞新音频累积）
        voice_ok, voiced_secs = has_enough_voice(combined_audio, sample_rate)

        if not voice_ok:
            status_msg = f"⚠️ 语音太短/太静({voiced_secs:.1f}s),已跳过"
            html_update = gr.update()  # 不更新HTML
        else:
            success = process_audio_chunk(combined_audio, sample_rate)
            
            if success:
                status_msg = f"✓ 已处理第 {state.processed_chunks} 段音频 (语音{voiced_secs:.1f}s)"
                # 有新内容时刷新缓存并记录渲染时间,减少闪烁
                transcript_html = format_transcript_html(state.transcripts, state.mapping_done, state.speaker_mapping)
                state.cached_transcript_html = transcript_html
                state.last_rendered_version = state.transcript_version
                state.last_ui_render_time = now
                html_update = gr.update(value=transcript_html)
            else:
                status_msg = f"⚠️ 处理失败"
                html_update = gr.update()
        
        # 处理完成，释放标记
        state.is_processing = False
    else:
        # 没有新内容且距离上次渲染过短时,直接复用缓存以降低刷新频率
        need_render = (
            state.transcript_version != state.last_rendered_version
            or (now - state.last_ui_render_time) > 1.0
            or not state.cached_transcript_html
        )
        if need_render:
            transcript_html = format_transcript_html(state.transcripts, state.mapping_done, state.speaker_mapping)
            state.cached_transcript_html = transcript_html
            state.last_rendered_version = state.transcript_version
            state.last_ui_render_time = now
            html_update = gr.update(value=transcript_html)
        else:
            html_update = gr.update()  # 不变更,避免闪烁
    
    return (
        html_update,
        f"{len(state.transcripts)} 条对话",
        status_msg
    )

def apply_speaker_mapping(*args):
    """应用说话人身份映射"""
    state.speaker_mapping = {}

    allowed_roles = set(ROLE_OPTIONS)

    n = MAX_SPEAKERS
    labels = list(args[:n])
    roles = list(args[n:2*n])
    names = list(args[2*n:3*n])

    for speaker_label, role_cn, name in zip(labels, roles, names):
        if not speaker_label:
            continue
        if role_cn not in allowed_roles:
            role_cn = "其他"

        # 映射角色
        role_map = {
            "医生": "doctor",
            "护士": "nurse",
            "患者": "patient",
            "家属": "family",
            "陪诊": "family",
            "家属/陪诊": "family",
            "其他": "other"
        }
        role = role_map.get(role_cn, "other")
        
        state.speaker_mapping[speaker_label] = {
            "role": role,
            "name": name or speaker_label
        }
    
    state.mapping_done = True
    
    # 更新对话显示
    updated_html = format_transcript_html(state.transcripts, True, state.speaker_mapping)
    state.cached_transcript_html = updated_html
    state.last_rendered_version = state.transcript_version
    state.last_ui_render_time = time.time()
    
    return (
        updated_html,
        gr.update(interactive=True),  # 启用生成报告按钮
        gr.update(value="<div style='color: #4cd964;'>✓ 身份设置已应用,可以生成报告</div>")
    )

def reset_speaker_mapping():
    """重置说话人映射"""
    state.speaker_mapping = {}
    state.mapping_done = False
    
    # 重新生成默认映射
    mapping_data = create_speaker_mapping_ui(list(state.unique_speakers))
    
    return (
        mapping_data,
        format_transcript_html(state.transcripts, False, {}),
        gr.update(interactive=False),  # 禁用生成报告
        gr.update(value="提示:请重新设置说话人身份"),
        *build_empty_mapping_updates()
    )

def generate_report():
    """生成结构化报告"""
    if not state.mapping_done:
        return "❌ 请先完成说话人身份设置并点击\"应用设置\""
    
    # 收集患者和医生的对话
    patient_texts = []
    doctor_texts = []
    all_dialogue = []
    
    for item in state.transcripts:
        speaker_label = item['speaker']
        text = item['text']
        time_str = f"{format_time(item['start'])}-{format_time(item['end'])}"
        
        if speaker_label in state.speaker_mapping:
            role = state.speaker_mapping[speaker_label]['role']
            name = state.speaker_mapping[speaker_label]['name']
            
            all_dialogue.append(f"**{name}** ({time_str}): {text}")
            
            if role in ['patient', 'family']:
                patient_texts.append(text)
            elif role in ['doctor', 'nurse']:
                doctor_texts.append(text)
    
    # 生成报告
    report = f"""
# 📋 问诊结构化报告

---

## 📌 病人基本信息
- **就诊时间**: {state.start_time.strftime('%Y-%m-%d %H:%M:%S') if state.start_time else '未知'}
- **问诊时长**: {format_time(state.transcripts[-1]['end']) if state.transcripts else '00:00'}
- **参与人数**: {len(state.unique_speakers)}人

---

## 🗣️ 病人自述
{' '.join(patient_texts) if patient_texts else '无'}

---

## 🩺 医生问诊摘要
{' '.join(doctor_texts) if doctor_texts else '无'}

---

## 💬 完整对话记录
{chr(10).join(all_dialogue)}

---

## 🔍 初步诊断与建议
> *待医生补充...*

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report

# ==================== Gradio界面构建 ====================

with gr.Blocks(
    title="智能医生问诊AI系统",
) as demo:
    
    # 标题
    gr.HTML("""
    <div class="header">
        <h1 style="margin: 0; display: flex; align-items: center; gap: 12px; font-size: 24px;">
            🩺 智能医生问诊AI系统
        </h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">
            基于语音识别和说话人分离的医疗问诊记录系统
        </p>
    </div>
    """)
    
    with gr.Row():
        # 左侧:问诊对话
        with gr.Column(scale=1):
            gr.Markdown("### 📝 问诊对话")
            
            # 控制按钮
            with gr.Row():
                start_btn = gr.Button("▶️ 开始问诊", variant="primary", size="lg")
                pause_btn = gr.Button("⏸ 暂停", interactive=False)
                stop_btn = gr.Button("⏹ 结束问诊", interactive=False, variant="stop")
            
            # 状态指示器
            status_box = gr.HTML(
                "<div style='color: #666; text-align: center; padding: 10px;'>就绪</div>",
                elem_classes="status-box"
            )
            
            # 音频输入 - 用于实时录音
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="录音",
                show_label=False,
                visible=False  # 隐藏,自动触发
            )
            
            # 对话转录窗口
            with gr.Row():
                gr.Markdown("**实时对话转录**")
                transcript_counter = gr.Markdown("0 条对话")
            
            # 音频状态提示
            audio_status = gr.Markdown("等待开始录音...", elem_classes="status-box")
            
            transcript_display = gr.HTML(
                value="<div style='color: #999; text-align: center; padding: 20px;'>点击\"开始问诊\"后开始录音</div>"
            )
            
            # 说话人身份设置
            gr.Markdown("### 👤 说话人身份设置")
            mapping_hint = gr.Markdown(
                "提示:问诊结束后,请为检测到的说话人分配身份。设置为\"医生\"或\"护士\"身份的消息将显示在右侧,其他身份消息显示在左侧。"
            )

            # 自定义下拉控件行
            label_boxes = []
            role_dropdowns = []
            name_boxes = []
            for i in range(MAX_SPEAKERS):
                with gr.Row():
                    lbl = gr.Textbox(label=f"说话人{i+1}", interactive=False, visible=False)
                    role = gr.Dropdown(choices=ROLE_OPTIONS, value="患者", label="角色", visible=False)
                    name = gr.Textbox(label="姓名", visible=False)
                label_boxes.append(lbl)
                role_dropdowns.append(role)
                name_boxes.append(name)
    
            with gr.Row():
                reset_mapping_btn = gr.Button("🔄 重置设置", variant="secondary", interactive=False)
                apply_mapping_btn = gr.Button("✓ 应用设置", variant="primary", interactive=False)
            
            generate_report_btn = gr.Button(
                "📋 生成结构化报告",
                variant="primary",
                interactive=False,
                size="lg"
            )
        
        # 右侧:结构化报告
        with gr.Column(scale=1):
            gr.Markdown("### 📄 结构化报告")
            report_display = gr.Markdown(
                value="""
---
请先完成以下步骤:
1. 点击"开始问诊"进行录音
2. 结束问诊后设置说话人身份
3. 点击"应用设置"
4. 点击"生成结构化报告"
---
                """
            )
    
    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
### 操作流程
1. **开始问诊**: 点击"开始问诊"按钮,系统自动开始录音
2. **实时转录**: 系统每10秒自动处理一次音频并显示转录结果
3. **暂停/继续**: 可随时暂停或继续录音
4. **结束问诊**: 点击"结束问诊",系统会自动合并相邻同一说话人的对话
5. **身份映射**: 为每个检测到的说话人设置角色和姓名
6. **应用设置**: 点击后对话窗口会按身份重新排列(医院方右侧,患者方左侧)
7. **生成报告**: 生成包含完整信息的结构化医疗报告

### 说话人角色说明
- **医生/护士**: 医院方人员,对话显示在右侧(蓝色)
- **患者/家属/陪诊**: 患者方人员,对话显示在左侧(绿色)
- **其他**: 其他参与者

### 技术特点
- ✅ 实时语音识别
- ✅ 自动说话人分离
- ✅ 智能对话合并
- ✅ 结构化报告生成
        """)
    
    # ==================== 事件绑定 ====================
    
    # 开始问诊 - 触发录音开始
    def on_start():
        updates = start_consultation()
        base = list(updates)
        # 打开音频输入组件
        base[12] = gr.update(visible=True)
        base += list(build_empty_mapping_updates())
        return tuple(base)
    
    mapping_state = gr.State()

    start_click = start_btn.click(
        fn=on_start,
        outputs=[
            start_btn, pause_btn, stop_btn, status_box,
            transcript_display,
            mapping_hint, apply_mapping_btn, reset_mapping_btn,
            generate_report_btn, report_display, transcript_counter,
            audio_status, audio_input,
            mapping_state
        ]
        + label_boxes
        + role_dropdowns
        + name_boxes
    )
    
    # 音频流处理 - 实时转录(累积到足够长度)
    audio_input.stream(
        fn=on_audio_stream,
        inputs=[audio_input],
        outputs=[transcript_display, transcript_counter, audio_status]
    )
    
    # 暂停/继续
    pause_btn.click(
        fn=pause_consultation,
        outputs=[status_box, pause_btn]
    )
    
    # 结束问诊
    stop_btn.click(
        fn=stop_consultation,
        outputs=[
            start_btn, pause_btn, stop_btn, status_box,
            transcript_display, mapping_state,
            mapping_hint, apply_mapping_btn, reset_mapping_btn,
            generate_report_btn, transcript_counter, audio_status
        ]
    ).then(
        fn=build_mapping_updates,
        inputs=[mapping_state],
        outputs=label_boxes + role_dropdowns + name_boxes
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[audio_input]
    )
    
    # 应用身份映射
    apply_mapping_btn.click(
        fn=apply_speaker_mapping,
        inputs=label_boxes + role_dropdowns + name_boxes,
        outputs=[transcript_display, generate_report_btn, status_box]
    )
    
    # 重置映射
    reset_mapping_btn.click(
        fn=reset_speaker_mapping,
        outputs=[
            mapping_state,
            transcript_display,
            generate_report_btn,
            mapping_hint,
        ] + label_boxes + role_dropdowns + name_boxes
    )
    
    # 生成报告
    generate_report_btn.click(
        fn=generate_report,
        outputs=[report_display]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .header {
            background: linear-gradient(to right, #1e90ff, #1a7feb);
            color: white;
            padding: 20px 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .status-box {
            padding: 12px;
            border-radius: 8px;
            background: #f5f5f5;
            text-align: center;
            font-weight: 500;
        }
        """
    )
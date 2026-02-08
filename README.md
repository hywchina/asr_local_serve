## 1. 需求描述


需求
1. 问诊流程重构
实时对话模拟：模拟医生问诊过程的对话流转，支持开始、暂停、停止控制
场景化问诊：提供多种预定义问诊场景（头痛症状、多人会诊、家属陪诊等）
渐进式身份识别：问诊过程中只显示说话人ID，问诊结束后才进行身份映射
2. 说话人身份管理系统
动态身份映射：问诊结束后自动检测说话人数量，生成可编辑的身份映射表单
多次修改支持：身份映射可随时调整，不再限于一次性设置
智能布局调整：身份设置后，对话界面自动按角色重新布局（医院人员右侧，患者相关人员左侧）
3. 结构化报告生成
条件触发机制：仅在身份映射完成后才能生成报告
智能内容提取：根据身份映射自动分类提取患者自述、医生问诊摘要等内容
专业报告格式：包含病人基本信息、症状描述、诊断建议等标准化章节


实时转录功能：
1. 点击开始问诊后，实时对话转录窗口是实时转录，但是此时是不知道这些人的身份的，因为会话时，可能是多人对话，可能有多个医生，多个病人或陪诊人员，因为实时对话转录窗口 在没有结束问诊时，都应该用说话人id来代替，
2. 当点击“结束问诊”时，前端页面可以对说话人身份id进行映射，即说话人1: 是王医生，说话人2:患者1 等等设定，设定完成之后，实时对话转录窗口的说话人id就要转变成设定的身份了。
3. 在点击“结束问诊”前，所有说话人说话内容均放在左侧。在设置了说话人身份了之后，所有医院人员（医生护士等）放到对话的右侧（类似医生视角的聊天窗口），所有病人相关人员聊天内容放在左侧。
4. 逻辑上时间顺序是：问诊结束后，共识别出几个人会话，然后在说话人设置的为止自动添加说话人，而不是事先就知道有几个说话人；同时结构报告，也是在设定说话人身份后，点击生成，才会生成的；
5. api服务代码不动，只通过前端代码，即可做到伪实时，比如说，前端录音，进行固定时间切分，逐个段落进行请求后端api，同时前端吧相邻同一人说话内容进行合并展示，这样不就好了吗


### 阶段1： 初始状态
前端：
  1. 主要的三个按钮：开始问诊、暂停、结束问诊
  2. 实时对话转录：聊天窗口，初始时是没有任何内容
  3. 模拟场景删除掉
  4. 会话人身份设置：初始时为空，设置提示词（提示：问诊结束后，请为检测到的说话人分配身份。设置为"医生"或"护士"身份的消息将显示在右侧，其他身份消息显示在左侧。），提供重置设置 和 应用设置
  5. 生成结构化报告：按钮置灰
后端：
  无操作

### 阶段 2：问诊进行中（伪实时）
前端：
  1. 点击“开始问诊”后，即开始录音
  2. “实时对话转录窗口”是实时转录，但是此时是不知道这些人的身份的，因为会话时，可能是多人对话，可能有多个医生，多个病人或陪诊人员，因为实时对话转录窗口 在没有结束问诊时，都应该用说话人id来代替，
  3. 通过前端代码，实现伪实时功能，即，不要等待全部录音结束再调用后端api，而是在前端录音过程中，进行固定时间切分，分批去请求后端api，将每次请求回来的api内容，即可显示在“实时对话转录”窗口，视觉上达到伪实时的效果。此时左右会话内容，均显示在“实时对话转录”窗口的左侧。

后端：
  1. 后端api服务是接受一段录音，然后返回一个json（包含说话人id，时间段，说话文本），将json返回给前端代码；
  
### 阶段 3：暂停
前端：点击【暂停】，暂停录音
后端：停止数据传输，避免暂用资源
  
### 阶段 4：问诊结束（一次性）
前端：
  1. 点击【结束问诊】，结束录音
  2. 把相邻同一人说话内容进行合并展示，例如 说话人1: 时间段【a，b】说话内容xxx，说话人1:时间段【b，c】，说话内容yyy，合并后 说话人1:时间段【a，c】，说话内容 xxxyyy。
  3. 统计一共有多少说话人数n，在“说话人身份设置”显示n个身份映射行，可以对说话人身份id进行映射，身份分两大类：医院（医生、护士、等），病人（患者、陪诊等）。即说话人1: 是王医生，说话人2:患者1 等等设定，设定完成之后，实时对话转录窗口的说话人id就要转变成设定的身份了。 同时在，在设置了说话人身份了之后，所有医院人员（医生护士等）放到对话的右侧（类似医生视角的聊天窗口），所有病人相关人员聊天内容放在左侧。
  4. 点击“生成结构化报告”，即将相关信息放到结构化报告模板中（待开发）




## 2. 环境配置

注意这个模型文件的配置，需要将其他几个本地模型的路径放在该模型配置中
/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common/configuration.json

ASR_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/Fun-ASR-Nano-2512"
VAD_MODEL_DIR = "/home/huyanwei/projects/llm_cache/ms/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common"
SV_MODEL_DIR  = "/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_sv_zh-cn_16k-common"


1. FunAudioLLM/Fun-ASR-Nano-2512
2. iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
3. iic/speech_campplus_speaker-diarization_common
4. iic/speech_campplus-transformer_scl_zh-cn_16k-common
5. iic/speech_campplus_sv_zh-cn_16k-common

modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./models/Fun-ASR-Nano-2512
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir ./models/speech_fsmn_vad_zh-cn-16k-common-pytorch
modelscope download --model iic/speech_campplus_speaker-diarization_common --local_dir ./models/speech_campplus_speaker-diarization_common
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./models/speech_campplus_sv_zh-cn_16k-common
modelscope download --model iic/speech_campplus-transformer_scl_zh-cn_16k-common --local_dir ./models/speech_campplus-transformer_scl_zh-cn_16k-common



## 3. 测试 


### 测试asr：
curl -X POST http://localhost:8000/asr \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
  -F "hotwords=开放时间"


### 测试sd ，jq 是json格式化作用
curl -s -X POST "http://127.0.0.1:8001/diarization" \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
| jq



### 测试asr_sd 服务https://modelscope.cn/papers/
curl -X POST http://localhost:8002/asr_sd \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
| jq

curl -X POST 'http://localhost:8002/asr_sd?debug_similarity=true' -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" | jq

curl -X POST 'http://localhost:8002/asr_sd' -F "file=@/home/huyanwei/projects/asr_local_serve/data/n_peoples_sample.mp3" | jq


curl -X POST "http://127.0.0.1:8002/asr_sd?session_id=clinic_001" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/3peoples.mp3" |jq


## 4. debug 记录

阅读前后端代码，然后辅助我完善代码

前端代码：/home/huyanwei/projects/asr_local_serve/services/gradio_asr_sd_demo.py
后端代码：/home/huyanwei/projects/asr_local_serve/services/local_asr_sd_serve.py
说明文档：/home/huyanwei/projects/asr_local_serve/README.md

问题：
1. 前端录音会缺失 WARNING:SpeechEngine:SD 处理失败（音频过短等原因）: modelscope error: The effective audio duration is too short.
2. 身份分两大类：医院（医生、护士、等），病人（患者、陪诊等）,UI上需要使用下拉框或者其他形式工用户选择，而不是手动填写
3. 实时转录窗口中，对话内容出现会在最低下出现，需要手动滑倒最底下才能看到最新的内容，需要类似聊天窗口那种呈现形式，最新的转录内容要能在窗口内显示，避免人工翻阅


4. 这是从1 数到100（100秒） 的录音转换结果，感觉切的太零散了
[1.6s-3.5s] speaker_1: 一二。

[3.8s-4.5s] unknown: 三。

[4.8s-5.5s] unknown: 是。

[6.8s-8.8s] speaker_1: 六七


5. 当前端一段时间没有声音时，后端api 会提示：WARNING:SpeechEngine:SD 处理失败（音频过短等原因）: modelscope error: The effective audio duration is too short.前端代码会提示：
Sending 31.0s audio to backend...
Received 0 segments

但是继续说话后，就不能继续转录了，前端ui提示：语音有效时长不足，继续累积中...，但是实际上已经有声音再说话了


6. 仔细检查代码逻辑，梳理语音处理顺序，避免有缺失、遗漏处理的情况，现在还是能发现：长时间没人说话情况下，再次说话的声音没有被识别转文字

7. 0208_01 
现在的问题是，当服务同时被多个session_id 访问时，只有一个session_id 的 对话内容会调取api，其他的都需要等待，即便是时间最早的那个session_id 结束了，其他的session_id 也不会成功调取后端api，查一下前后端逻辑代码。我需要的效果是：能保证多个session_id 能同时访问服务并得到响应

8. 0208_02
现在两个会话窗口是能有后端反馈的，但是开到第三个窗口（会话）时，第三个会话就是反应特别慢，等很久才有反馈，并且有遗漏，我需要的均衡一点的，类似消息队列一样，均衡一点，同时保证所有会话接收到的语音都被识别到，而不是被遗漏
# 测试asr：
curl -X POST http://localhost:8000/asr \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
  -F "hotwords=开放时间"


# 测试sd ，jq 是json格式化作用
curl -s -X POST "http://127.0.0.1:8001/diarization" \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
| jq



# 测试asr_sd 服务https://modelscope.cn/papers/
curl -X POST http://localhost:8002/asr_sd \
  -F "file=@/home/huyanwei/projects/asr_local_serve/data/speakers_example.wav" \
| jq




# 本地模型配置
注意这个模型文件的配置，需要将其他几个本地模型的路径放在该模型配置中
/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common/configuration.json



## 需求
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



## prompt1 

实时转录功能：
1. 点击开始问诊后，实时对话转录窗口是实时转录，但是此时是不知道这些人的身份的，因为会话时，可能是多人对话，可能有多个医生，多个病人或陪诊人员，因为实时对话转录窗口 在没有结束问诊时，都应该用说话人id来代替，
2. 当点击“结束问诊”时，前端页面可以对说话人身份id进行映射，即说话人1: 是王医生，说话人2:患者1 等等设定，设定完成之后，实时对话转录窗口的说话人id就要转变成设定的身份了。
3. 在点击“结束问诊”前，所有说话人说话内容均放在左侧。在设置了说话人身份了之后，所有医院人员（医生护士等）放到对话的右侧（类似医生视角的聊天窗口），所有病人相关人员聊天内容放在左侧。
4. 逻辑上时间顺序是：问诊结束后，共识别出几个人会话，然后在说话人设置的为止自动添加说话人，而不是事先就知道有几个说话人；同时结构报告，也是在设定说话人身份后，点击生成，才会生成的；

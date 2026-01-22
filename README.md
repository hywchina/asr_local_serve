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

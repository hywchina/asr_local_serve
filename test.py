# 版本要求 modelscope version 升级至最新版本 funasr 升级至最新版本
from modelscope.pipelines import pipeline
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='/home/huyanwei/projects/llm_cache/ms/model/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)
input_wav = '/home/huyanwei/projects/asr_local_serve/data/n_peoples_sample.mp3'
result = sd_pipeline(input_wav)
print(result)
# 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
result = sd_pipeline(input_wav, oracle_num=2)
print(result)
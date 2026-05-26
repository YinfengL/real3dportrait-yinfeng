import os
import threading
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, HubertModel

# 全局变量
hubert_model_dir = '/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8'
wav2vec2_processor = None
hubert_model = None
_model_lock = threading.Lock()  # 线程锁，防止并发初始化冲突

def get_hubert_from_16k_wav(wav_16k_name):
    """读取音频并调用 HuBERT 提取特征"""
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model, wav2vec2_processor
    
    # 确保模型只加载一次且线程安全
    if hubert_model is None:
        with _model_lock:
            if hubert_model is None:
                print(">>> [HuBERT] Loading model and processor...")
                if os.path.exists(hubert_model_dir):
                    # 在初始化时一次性加载到 GPU 并保持在评估模式
                    hubert_model = HubertModel.from_pretrained(hubert_model_dir).to(device).eval()
                    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(hubert_model_dir)
                else:
                    raise FileNotFoundError(f"Hubert model directory not found at {hubert_model_dir}")

    # 数据预处理
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    
    # 获取输入值
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    input_values_all = input_values_all.to(device)

    # HuBERT 逻辑处理：将超长音频切片处理以避免显存溢出
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    
    for i in range(num_iter):
        start_idx = 0 if i == 0 else clip_length * i
        end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        
        # 使用预加载好的 hubert_model
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
        
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all

    if input_values.shape[1] >= kernel:
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
        
    ret = torch.cat(res_lst, dim=0).cpu() # 将结果转回 CPU

    # 补齐逻辑
    if ret.shape[0] < expected_T:
        padding = expected_T - ret.shape[0]
        ret = torch.cat([ret, ret[-1:].repeat(padding, 1)], dim=0)
    else:
        ret = ret[:expected_T]

    return ret

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--video_id', type=str, default='May', help='')
    args = parser.parse_args()
    
    person_id = args.video_id
    wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
    hubert_npy_name = f"data/processed/videos/{person_id}/aud_hubert.npy"
    
    speech_16k, _ = sf.read(wav_16k_name)
    hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    np.save(hubert_npy_name, hubert_hidden.detach().numpy())
    print(f"Saved at {hubert_npy_name}")

import os, sys
import traceback
import threading
from datetime import datetime
import argparse
import gradio as gr
import random, string
import torch

# 确保项目根目录在系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inference.real3d_infer import GeneFace2Infer
tts_engine = None 

# 全局模型实例
infer_obj = None
tts_engine = None
_tts_lock = threading.Lock()

def normalize_path(path):
    if not path or path == '': return ''
    if not os.path.isabs(path): path = os.path.abspath(path)
    return os.path.normpath(path)

class Inferer(GeneFace2Infer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get('device', 'cuda:0')
        print(f"[*] Pre-warming models on {self.device}...")
        
        # 【关键修复】自动遍历对象中所有的 PyTorch 模型实例并移动到显存
        # 这样不管父类怎么命名模型，只要它是 torch.nn.Module 都会被移动
        with torch.no_grad():
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                # 如果是 PyTorch 模型，将其移动到设备
                if isinstance(attr, torch.nn.Module):
                    try:
                        attr.to(self.device)
                        print(f"    - Moved {attr_name} to {self.device}")
                    except:
                        pass


    def infer_once_args(self, *args, custom_out_path=None, **kargs):
        keys = [
            'src_image_name', 'drv_audio_name', 'drv_pose_name', 'bg_image_name',
            'blink_mode', 'temperature', 'mouth_amp', 'out_mode', 'map_to_init_pose',
            'low_memory_usage', 'hold_eye_opened', 'a2m_ckpt', 'head_ckpt', 'torso_ckpt', 'min_face_area_percent'
        ]
        
        inp = {keys[i]: args[i] for i in range(len(keys))}
        
        inp['a2m_ckpt'] = normalize_path(inp['a2m_ckpt'])
        inp['head_ckpt'] = normalize_path(inp['head_ckpt'])
        inp['torso_ckpt'] = normalize_path(inp['torso_ckpt'])

        if custom_out_path:
            inp['out_name'] = custom_out_path
            os.makedirs(os.path.dirname(custom_out_path), exist_ok=True)
        else:
            out_dir = "infer_out/tmp"
            os.makedirs(out_dir, exist_ok=True)
            rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            inp['out_name'] = os.path.join(out_dir, f"{os.path.basename(inp['src_image_name'])}_{rand_str}.mp4")
        
        inp['seed'] = 42
        
        # 使用 torch.no_grad 保证推理过程是只读的，适合并发
        with torch.no_grad():
            return self.infer_once(inp)

def process_with_tts(
    src_image, drv_audio, drv_pose, bg_image,
    blink_mode, temperature, mouth_amp, out_mode,
    map_to_init_pose, low_memory_usage, hold_eye_opened,
    a2m_ckpt, head_ckpt, torso_ckpt, min_face_area,
    text_input=None, custom_output_path=None
):
    global infer_obj, tts_engine
    
    # TTS 为纯 CPU 任务，可以并发，但受限于 PaddleTTS 引擎，加锁以防万一
    final_audio_path = drv_audio
    if text_input and len(text_input.strip()) > 0:
        with _tts_lock:
            if tts_engine is None:
                tts_engine = PaddleTTS()
            temp_audio = f"temp_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{random.randint(1000,9999)}.wav"
            final_audio_path = tts_engine.generate_audio(text_input, output_path=temp_audio)
    
    if infer_obj is None:
        raise RuntimeError("Inferer not initialized")
        
    # 直接调用推理
    return infer_obj.infer_once_args(
        src_image, final_audio_path, drv_pose, bg_image,
        blink_mode, temperature, mouth_amp, out_mode,
        map_to_init_pose, low_memory_usage, hold_eye_opened,
        a2m_ckpt, head_ckpt, torso_ckpt, min_face_area,
        custom_out_path=custom_output_path
    ), None

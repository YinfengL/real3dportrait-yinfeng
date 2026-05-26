import os
import asyncio
import subprocess
from datetime import datetime

class EdgeTTS:
    def __init__(self):
        self.voice = "zh-CN-XiaoxiaoNeural"
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_out/tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_audio(self, text, output_path=None):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("输入文本不能为空")

        if output_path is None:
            filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
            output_path = os.path.join(self.cache_dir, filename)
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self.cache_dir, output_path)
        
        if not output_path.endswith('.wav'):
            output_path = output_path.rsplit('.', 1)[0] + '.wav'

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._generate_async(text, output_path))
        finally:
            loop.close()

        if os.path.exists(output_path):
            return os.path.abspath(output_path)
        else:
            raise FileNotFoundError(f"TTS 未生成文件: {output_path}")

    async def _generate_async(self, text, output_wav_path):
        temp_mp3 = output_wav_path.replace('.wav', '.mp3')
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(temp_mp3)
            
            cmd = [
                'ffmpeg', '-y', '-i', temp_mp3, 
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', 
                output_wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception as e:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            raise e

# 🔥🔥🔥 关键兼容层：让旧代码导入 PaddleTTS 时也能拿到 EdgeTTS
PaddleTTS = EdgeTTS

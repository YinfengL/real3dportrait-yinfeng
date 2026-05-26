import os
import sys
import numpy as np
import uvicorn
import asyncio
import traceback
import re
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import subprocess

# =========================
# 基础环境
# =========================

devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, sys.stderr.fileno())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(title="Real3D-Portrait TTS API")

# TTS / CPU 线程池
executor = ThreadPoolExecutor(max_workers=8)

# 全局变量
gpu_semaphore = None
infer_obj = None
tts_engine = None
static_data_cache = {}

A2M_CKPT = os.path.join(
    PROJECT_ROOT,
    'checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt'
)
HEAD_CKPT = ''
TORSO_CKPT = os.path.join(
    PROJECT_ROOT,
    'checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt'
)

# 窗口大小：允许最多 N 个 chunk 进入“活跃调度”
WINDOW_SIZE = 3


@app.on_event("startup")
async def startup_event():
    global gpu_semaphore, infer_obj, tts_engine

    gpu_semaphore = asyncio.Semaphore(WINDOW_SIZE)

    print("Loading Real3D-Portrait models...", flush=True)
    from inference.app_real3dportrait import Inferer, normalize_path

    infer_obj = Inferer(
        audio2secc_dir=normalize_path(A2M_CKPT),
        head_model_dir=normalize_path(HEAD_CKPT) if HEAD_CKPT else '',
        torso_model_dir=normalize_path(TORSO_CKPT),
        device='cuda:0'
    )

    import inference.app_real3dportrait as app_module
    app_module.infer_obj = infer_obj

    try:
        from tts_engine import EdgeTTS
        tts_engine = EdgeTTS()
        print("Edge-TTS Loaded", flush=True)
    except ImportError:
        print("Edge-TTS not found, falling back to PaddleTTS...", flush=True)
        from tts_engine import PaddleTTS
        tts_engine = PaddleTTS()

    app_module.tts_engine = tts_engine

    try:
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        import soundfile as sf
        dummy_wav = np.zeros(16000, dtype=np.float32)
        sf.write('/tmp/dummy_hubert.wav', dummy_wav, 16000)
        _ = get_hubert_from_16k_wav('/tmp/dummy_hubert.wav')
        print("HuBERT Pre-loaded", flush=True)
    except Exception as e:
        print(f"[HuBERT] Pre-load skipped: {e}", flush=True)

    print("Models and TTS loaded", flush=True)


from inference.app_real3dportrait import normalize_path


def get_video_duration_seconds(file_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return 0.0
    except Exception:
        return 0.0


def to_abs_path(p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)


def write_manifest(task_dir, task_id, generated_files, status="running", error=None):
    manifest = {
        "task_id": task_id,
        "status": status,
        "generated_files": generated_files,
        "error": error,
    }
    manifest_path = os.path.join(task_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


class GenerationRequest(BaseModel):
    src_image_path: str
    text_input: Optional[str] = None
    drv_audio_path: Optional[str] = None
    drv_pose_path: Optional[str] = "data/raw/examples/May_5s.mp4"
    bg_image_path: Optional[str] = ""
    blink_mode: str = 'period'
    temperature: float = 0.2
    mouth_amp: float = 0.45
    out_mode: str = 'final'
    min_face_area_percent: float = 0.2


def split_text_smart(text: str) -> List[str]:
    parts = re.split(r'([，。！？；：\.\!\?\;\,\(\)\（\）])', text)
    segments = []
    current = ""

    FAST_CHUNK_SIZE = 25
    NORMAL_CHUNK_SIZE = 25
    FAST_CHUNK_COUNT = 3
    chunk_index = 0

    for i in range(0, len(parts), 2):
        chunk = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        clean_chunk = re.sub(r'[，。！？；：\.\!\?\;\,\(\)\（\） \n\t]', '', chunk)
        if not clean_chunk:
            continue

        max_len = FAST_CHUNK_SIZE if chunk_index < FAST_CHUNK_COUNT else NORMAL_CHUNK_SIZE

        if len(current) + len(chunk) <= max_len:
            current += chunk
        else:
            if current:
                segments.append(current)
                chunk_index += 1
            current = chunk

    if current:
        clean_current = re.sub(r'[，。！？；：\.\!\?\;\,\(\)\（\） \n\t]', '', current)
        if clean_current:
            segments.append(current)

    return segments


def get_or_create_static_data(src_image, drv_pose, bg_image, min_face_area):
    import hashlib
    cache_key = hashlib.md5(f"{src_image}_{drv_pose}_{bg_image}_{min_face_area}".encode()).hexdigest()

    if cache_key in static_data_cache:
        print(f"[STATIC CACHE HIT] Key: {cache_key[:8]}", flush=True)
        return static_data_cache[cache_key]

    try:
        print(f"[STATIC CACHE MISS] Preprocessing static data for key: {cache_key[:8]}...", flush=True)
        static_sample = infer_obj.prepare_static_batch(
            src_image_path=src_image,
            drv_pose_path=drv_pose,
            bg_image_path=bg_image,
            min_face_area_percent=min_face_area
        )
        static_data_cache[cache_key] = static_sample
        print(f"[STATIC CACHE SAVED] Key: {cache_key[:8]}", flush=True)
        return static_sample
    except Exception as e:
        print(f"[ERROR] prepare_static_batch failed: {e}", flush=True)
        traceback.print_exc()
        return None


def run_inference_sync_impl(index, audio_path, base_params, static_sample, work_dir):
    filename = f"part_{index:04d}.mp4"
    full_path = os.path.join(work_dir, filename)

    try:
        from inference.infer_utils import save_wav16k
        import torch
        import random
        import time

        wav16k_name = save_wav16k(audio_path)
        hubert_np = infer_obj.get_hubert(wav16k_name)
        f0_torch = infer_obj.get_f0(wav16k_name)

        if isinstance(f0_torch, torch.Tensor):
            f0_np = f0_torch.cpu().numpy()
        else:
            f0_np = f0_torch

        if f0_np.shape[0] > len(hubert_np):
            f0_np = f0_np[:len(hubert_np)]
        else:
            num_to_pad = len(hubert_np) - f0_np.shape[0]
            f0_np = np.pad(f0_np, pad_width=((0, num_to_pad), (0, 0)))

        t_x = hubert_np.shape[0]
        hubert_tensor = torch.from_numpy(hubert_np).float().unsqueeze(0)
        f0_tensor = torch.from_numpy(f0_np).float().reshape([1, -1])

        dynamic_inp = {
            'mouth_amp': base_params.get('mouth_amp', 0.45),
            'blink_mode': base_params.get('blink_mode', 'period'),
            'map_to_init_pose': True,
            'temperature': base_params.get('temperature', 0.2)
        }

        static_copy = dict(static_sample)

        batch = infer_obj.prepare_dynamic_batch(
            static_sample=static_copy,
            hubert=hubert_tensor,
            f0=f0_tensor,
            t_x=t_x,
            inp=dynamic_inp
        )

        seed = int(time.time()) + index
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        fake_inp = {
            'drv_audio_name': wav16k_name,
            'out_name': full_path,
            'blink_mode': base_params.get('blink_mode', 'period'),
            'mouth_amp': base_params.get('mouth_amp', 0.45),
            'map_to_init_pose': True,
            'temperature': base_params.get('temperature', 0.2),
            'seed': seed
        }

        out_fname = infer_obj.forward_system(batch, fake_inp)

        if not out_fname or not os.path.exists(out_fname):
            raise RuntimeError(f"forward_system returned invalid path: {out_fname}")

        return out_fname

    except Exception as e:
        print(f"[Sync Worker {index}] Error: {e}", flush=True)
        traceback.print_exc()
        raise


async def generate_chunk_worker(index: int, chunk_text: str, base_params: dict, static_sample: dict, work_dir: str, start_time: datetime):
    audio_path = None
    loop = asyncio.get_event_loop()

    print(f"[Worker {index}] Received Text: \"{chunk_text}\"", flush=True)

    try:
        from inference.app_real3dportrait import tts_engine as local_tts

        audio_path = await loop.run_in_executor(
            executor,
            lambda: local_tts.generate_audio(chunk_text)
        )

        async with gpu_semaphore:
            print(f"Chunk {index} starting inference...", flush=True)
            video_res = run_inference_sync_impl(index, audio_path, base_params, static_sample, work_dir)

            duration = (datetime.now() - start_time).total_seconds()
            print(f"[Worker {index}] Done Chunk {index} in {duration:.2f}s", flush=True)

            return {
                "status": "success",
                "path": video_res,
                "index": index,
                "time": duration
            }

    except Exception as e:
        print(f"[Worker {index}] Failed: {e}", flush=True)
        traceback.print_exc()
        return {"status": "failed", "error": str(e), "index": index}

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                resample_path = audio_path.replace(".wav", "_16k.wav")
                if os.path.exists(resample_path):
                    os.remove(resample_path)
            except:
                pass


async def run_windowed_batch(chunks, base_params, static_sample, work_dir, start_time, task_id):
    results = []
    total = len(chunks)
    if total == 0:
        write_manifest(work_dir, task_id, [], status="completed", error=None)
        return results

    pending = set()
    next_index = 0
    generated_files = []

    while next_index < total and len(pending) < WINDOW_SIZE:
        task = asyncio.create_task(
            generate_chunk_worker(
                next_index, chunks[next_index], base_params, static_sample, work_dir, start_time
            )
        )
        pending.add(task)
        next_index += 1

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            try:
                r = task.result()
                results.append(r)

                if r.get("status") == "success":
                    generated_files.append(os.path.basename(r["path"]))
                    write_manifest(work_dir, task_id, generated_files=generated_files, status="running", error=None)
                else:
                    write_manifest(work_dir, task_id, generated_files=generated_files, status="running", error=r.get("error"))
            except Exception as e:
                results.append({"status": "failed", "error": str(e), "index": -1})
                write_manifest(work_dir, task_id, generated_files=generated_files, status="running", error=str(e))

        while next_index < total and len(pending) < WINDOW_SIZE:
            task = asyncio.create_task(
                generate_chunk_worker(
                    next_index, chunks[next_index], base_params, static_sample, work_dir, start_time
                )
            )
            pending.add(task)
            next_index += 1

    write_manifest(work_dir, task_id, generated_files=generated_files, status="completed", error=None)
    return results


async def run_batch_job(req: GenerationRequest, task_id: str, work_dir: str, start_time: datetime):
    try:
        if not req.text_input:
            write_manifest(work_dir, task_id, generated_files=[], status="failed", error="text_input is required")
            return

        chunks = split_text_smart(req.text_input)
        print(f"Text split into {len(chunks)} chunks.", flush=True)

        bg_path = to_abs_path(req.bg_image_path) if req.bg_image_path else ""
        src_image_full = to_abs_path(req.src_image_path)
        drv_pose_full = to_abs_path(req.drv_pose_path)

        print("[MAIN] Preparing static batch data...", flush=True)
        static_sample = get_or_create_static_data(
            src_image=src_image_full,
            drv_pose=drv_pose_full,
            bg_image=bg_path,
            min_face_area=req.min_face_area_percent
        )

        if static_sample is None:
            write_manifest(work_dir, task_id, generated_files=[], status="failed", error="Failed to prepare static data.")
            return

        base_params = {
            "src_image": src_image_full,
            "drv_pose": drv_pose_full,
            "bg_image": bg_path,
            "blink_mode": req.blink_mode,
            "temperature": req.temperature,
            "mouth_amp": req.mouth_amp,
            "out_mode": req.out_mode,
            "min_face_area": req.min_face_area_percent,
            "a2m_ckpt": normalize_path(A2M_CKPT),
            "head_ckpt": normalize_path(HEAD_CKPT) if HEAD_CKPT else "",
            "torso_ckpt": normalize_path(TORSO_CKPT)
        }

        print(f"[MAIN] Running sliding window batch, window={WINDOW_SIZE}", flush=True)

        results = await run_windowed_batch(
            chunks=chunks,
            base_params=base_params,
            static_sample=static_sample,
            work_dir=work_dir,
            start_time=start_time,
            task_id=task_id
        )

        sorted_results = sorted(results, key=lambda x: x.get('index', 10**9))
        success_paths = [r['path'] for r in sorted_results if r.get('status') == 'success']
        failed_tasks = [r for r in sorted_results if r.get('status') != 'success']

        if failed_tasks:
            print(f"[ERROR] Some chunks failed: {failed_tasks}", flush=True)
            write_manifest(
                work_dir,
                task_id,
                generated_files=[os.path.basename(p) for p in success_paths],
                status="failed",
                error=failed_tasks[0].get('error', 'unknown error')
            )
            return

        total_video_seconds = 0.0
        for path in success_paths:
            if os.path.exists(path):
                total_video_seconds += get_video_duration_seconds(path)

        write_manifest(
            work_dir,
            task_id,
            generated_files=[os.path.basename(p) for p in success_paths],
            status="completed",
            error=None
        )

        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"Batch Task {task_id} Completed!", flush=True)
        print(f"Total Time: {total_duration:.2f}s", flush=True)
        print(f"Generated Video Total Duration: {total_video_seconds:.2f}s ({total_video_seconds/60:.2f} mins)", flush=True)

    except Exception as e:
        print(f"[BATCH JOB ERROR] {e}", flush=True)
        traceback.print_exc()
        write_manifest(work_dir, task_id, generated_files=[], status="failed", error=str(e))


@app.get("/download/{task_id}/{filename}")
def download_video(task_id: str, filename: str):
    file_path = os.path.join(PROJECT_ROOT, "infer_out/batch_tasks", task_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)


@app.get("/task-status/{task_id}")
def task_status(task_id: str):
    task_dir = os.path.join(PROJECT_ROOT, "infer_out/batch_tasks", task_id)
    manifest_path = os.path.join(task_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/generate-batch")
async def generate_batch_video(req: GenerationRequest):
    start_time = datetime.now()
    task_id = start_time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(PROJECT_ROOT, "infer_out/batch_tasks", task_id)
    os.makedirs(work_dir, exist_ok=True)

    print(f"Batch Task {task_id} Starting...", flush=True)
    write_manifest(work_dir, task_id, generated_files=[], status="running", error=None)

    asyncio.create_task(run_batch_job(req, task_id, work_dir, start_time))

    return {
        "status": "started",
        "task_id": task_id
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

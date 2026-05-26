# real3dportrait-yinfeng

This repository is an engineering adaptation of [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) (ICLR 2024 Spotlight). It keeps the original Real3D-Portrait inference pipeline and adds a FastAPI HTTP service, Edge-TTS integration, long-text chunking, asynchronous task tracking, static feature caching, and chunk-based video generation.

The goal of this project is not to redesign the underlying talking-head generation model. Instead, it turns the original research demo into a service-oriented digital human video generation module. A typical pipeline is:

```text
Text input / RAG-generated answer
→ Edge-TTS audio generation
→ audio normalization and HuBERT feature extraction
→ Real3D-Portrait inference
→ talking portrait video output
→ task status query and file download
```

The short-text generation flow has been verified in an AutoDL Linux + NVIDIA GPU environment. A successful run should produce an output similar to:

```text
infer_out/batch_tasks/{task_id}/part_0000.mp4
```

---

## 1. Main Differences from the Upstream Project

| Aspect | Upstream Real3D-Portrait | This Project |
|---|---|---|
| Usage | Gradio WebUI / CLI inference | FastAPI HTTP API, while keeping the original inference entry |
| Input | Source image + driving audio / driving video | Text input through TTS; external audio input is also supported |
| Long text | No service-level chunking in the default inference flow | Sentence splitting, chunk-level generation and task-level manifest |
| Task execution | Demo / script-style execution | Background task execution with task ID, status query and file download |
| Concurrency | No GPU concurrency window in the original demo flow | Sliding-window chunk inference, default window size 3 |
| Static cache | Static visual preprocessing is more tightly coupled with dynamic inference | Reuses static features for the same portrait, pose and background |
| Positioning | Official research implementation and demo | Service-oriented digital human video generation module |

---

## 2. Recommended Runtime Environment

A Linux + NVIDIA GPU server is recommended, such as AutoDL or an equivalent cloud GPU environment. Local CPU or macOS environments are not recommended for first-time reproduction.

Please refer to the upstream installation files first:

```text
docs/prepare_env/install_guide.md
docs/prepare_env/requirements.txt
```

Recommended baseline:

```text
Python 3.9
PyTorch 2.0.1
CUDA 11.7
ffmpeg
pytorch3d
mmcv
FastAPI
Uvicorn
Edge-TTS
```

Typical setup:

```bash
conda create -n real3dportrait python=3.9
conda activate real3dportrait

conda install conda-forge::ffmpeg
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch3d::pytorch3d

pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0

pip install -r docs/prepare_env/requirements.txt -v
pip install fastapi uvicorn edge-tts
```

Check `ffmpeg`:

```bash
ffmpeg -version
```

> Note: this project has also been adapted in an AutoDL / CUDA 12.1 environment. If your CUDA version is different, check the include and library paths for custom CUDA operators in `modules/eg3ds/torch_utils/custom_ops.py`.

---

## 3. External Model Files and Placement

Large model files are not tracked by GitHub. They must be downloaded separately or copied from an existing working environment. The paths below are based on the verified runtime environment.

### 3.1 File Placement Summary

| File Type | Placement | Tracked by Git | Notes |
|---|---|---:|---|
| Real3D-Portrait checkpoints | `checkpoints/` | No | Contains `240210_real3dportrait_orig/` and `pretrained_ckpts/` |
| audio2secc checkpoint | `checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt` | No | Required |
| secc2plane torso checkpoint | `checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt` | No | Required |
| EG3D baseline checkpoint | `checkpoints/pretrained_ckpts/eg3d_baseline_run2/model_ckpt_steps_100000.ckpt` | No | Present in the verified environment; recommended to keep |
| MIT-B0 checkpoint | `checkpoints/pretrained_ckpts/mit_b0.pth` | No | Used by visual preprocessing / encoding components |
| BFM / 3DMM large files | `deep_3drecon/BFM/` | No | Download separately |
| BFM small metadata files | `deep_3drecon/BFM/select_vertex_id.mat`, etc. | Yes | Included in this repository |
| HuBERT | `/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/` | No | Download from HuggingFace or a mirror |
| Custom HuBERT location | Any local directory with `HUBERT_MODEL_DIR` | No | Example: `pretrained_models/hubert-large-ls960-ft/` |
| MediaPipe segmentation model | `data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite` | Can be tracked | Present in the verified environment |
| Generated videos | `infer_out/batch_tasks/{task_id}/` | No | Generated output; do not commit |

---

### 3.2 Real3D-Portrait Pretrained Checkpoints

Download the pretrained checkpoints from the upstream README and unzip/copy them under:

```text
checkpoints/
```

Download links:

- Google Drive: https://drive.google.com/drive/folders/1MAveJf7RvJ-Opg1f5qhLdoRoC_Gc6nD9?usp=sharing
- Baidu Netdisk: https://pan.baidu.com/s/1Mjmbn0UtA1Zm9owZ7zWNgQ?pwd=6x4f  
  password: `6x4f`

Verified structure:

```text
checkpoints/
├── 240210_real3dportrait_orig/
│   ├── audio2secc_vae/
│   │   ├── config.yaml
│   │   └── model_ckpt_steps_400000.ckpt
│   └── secc2plane_torso_orig/
│       ├── config.yaml
│       └── model_ckpt_steps_100000.ckpt
└── pretrained_ckpts/
    ├── eg3d_baseline_run2/
    │   ├── config.yaml
    │   └── model_ckpt_steps_100000.ckpt
    └── mit_b0.pth
```

During startup, warnings like the following may appear:

```text
base.yaml not exist
secc_img2plane_orig.yaml not exist
```

In the verified environment, these warnings did not block inference as long as the `.ckpt` and `config.yaml` files above existed.

Check commands:

```bash
cd /root/Real3DPortrait

find checkpoints -maxdepth 4 -type f | sort

ls -lh checkpoints/240210_real3dportrait_orig/audio2secc_vae/
ls -lh checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/
ls -lh checkpoints/pretrained_ckpts/
```

---

### 3.3 BFM / 3DMM Face Model

Download the BFM files from the links provided in the upstream README and place them under:

```text
deep_3drecon/BFM/
```

Download links:

- Google Drive: https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=sharing
- Baidu Netdisk: https://pan.baidu.com/s/1aqv1z_qZ23Vp2VP4uxxblQ?pwd=m9q5  
  password: `m9q5`

The verified environment contains:

```text
deep_3drecon/BFM/
├── 01_MorphableModel.mat              # download separately
├── 3DMM BFM.zip                       # downloaded package; can be kept as backup
├── BFM_exp_idx.mat                    # download separately
├── BFM_front_idx.mat                  # download separately
├── BFM_model_front.mat                # generated by script or provided by the package
├── Exp_Pca.bin                        # download separately
├── basel_53201.txt                    # present in the verified environment
├── facemodel_info.mat                 # download separately
├── index_mp468_from_mesh35709.npy     # included in this repository
├── index_mp468_from_mesh35709_v1.npy  # present in the verified environment
├── index_mp468_from_mesh35709_v2.npy  # present in the verified environment
├── index_mp468_from_mesh35709_v3.npy  # present in the verified environment
├── index_mp468_from_mesh35709_v3.1.npy# present in the verified environment
├── select_vertex_id.mat               # included in this repository
├── similarity_Lm3D_all.mat            # included in this repository
└── std_exp.txt                        # included in this repository
```

If `BFM_model_front.mat` is not included in the downloaded package, run the conversion script once before inference:

```bash
cd deep_3drecon/BFM
python -c "from util.load_mats import transferBFM09; transferBFM09('.')"
```

Check command:

```bash
cd /root/Real3DPortrait
ls -lh deep_3drecon/BFM/
```

---

### 3.4 HuBERT Audio Feature Model

This project uses HuBERT features during audio-driven inference. The required model is:

```text
facebook/hubert-large-ls960-ft
```

HuBERT is not included in this GitHub repository. If HuBERT is missing, the API service may still start and Edge-TTS may still load, but video generation will fail during chunk inference with an error similar to:

```text
Hubert model directory not found
```

#### Recommended Download Method

If the server can access HuggingFace or a HuggingFace mirror, run:

```bash
source /root/miniconda3/bin/activate real3dportrait
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

python - << 'PY'
from transformers import HubertModel, Wav2Vec2Processor

model_name = "facebook/hubert-large-ls960-ft"

print("Downloading processor...")
processor = Wav2Vec2Processor.from_pretrained(model_name)

print("Downloading model with safetensors...")
model = HubertModel.from_pretrained(model_name, use_safetensors=True)

print("HuBERT loaded successfully with safetensors.")
PY
```

After downloading, the default cache path is usually:

```text
/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/
```

The verified environment has a structure similar to:

```text
/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/
├── blobs/
├── refs/
└── snapshots/
    ├── 4e59ee873209637dcf3f545914f5b021375ebba1/
    │   └── model.safetensors
    └── ece5fabbf034c1073acae96d5401b25be96709d8/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        ├── pytorch_model.bin
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── vocab.json
```

Check commands:

```bash
find /root/.cache/huggingface/hub -type d -path "*models--facebook--hubert-large-ls960-ft*" | sort

find /root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots \
  \( -name "model.safetensors" -o -name "pytorch_model.bin" -o -name "config.json" -o -name "preprocessor_config.json" \) \
  2>/dev/null | sort
```

#### PyTorch Version Note

If loading `pytorch_model.bin` raises an error requiring `torch >= 2.6`, do not blindly upgrade PyTorch. Real3D-Portrait can be sensitive to the PyTorch / CUDA / mmcv / pytorch3d combination. Use `model.safetensors` instead:

```python
HubertModel.from_pretrained(model_name, use_safetensors=True)
```

#### Custom Local HuBERT Path

The current code can automatically search the HuggingFace cache and prioritizes snapshots containing `model.safetensors`. You can also specify a local HuBERT directory manually:

```bash
export HUBERT_MODEL_DIR=/path/to/hubert-large-ls960-ft
```

Example:

```bash
export HUBERT_MODEL_DIR=/root/Real3DPortrait/pretrained_models/hubert-large-ls960-ft
```

If you place HuBERT under `pretrained_models/`, do not commit it to Git.

---

### 3.5 MediaPipe Segmentation Model

In the verified environment, the MediaPipe segmentation model is located at:

```text
data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite
```

If the file is missing, download it manually:

```bash
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

Place it under:

```text
data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite
```

Check command:

```bash
find . \( -name "*.tflite" -o -iname "*selfie*" -o -iname "*segment*" \) 2>/dev/null | sort
```

---

## 4. Output Directory

Task outputs are saved under:

```text
infer_out/batch_tasks/{task_id}/
```

In the verified AutoDL environment, `infer_out` is a symlink:

```text
infer_out -> /root/autodl-tmp/Real3DPortrait_outputs
```

The runtime structure is:

```text
infer_out/
├── batch_tasks/
├── tmp/
└── tts_cache/
```

If the repository is newly cloned, initialize the output directories first:

```bash
mkdir -p infer_out/tts_cache infer_out/batch_tasks infer_out/tmp
```

If you use the AutoDL symlink setup, create the real target directories:

```bash
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/tts_cache
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/batch_tasks
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/tmp
```

Check commands:

```bash
ls -lh infer_out
ls -lh infer_out/
ls -lt infer_out/batch_tasks | head
```

---

## 5. Quick Start

Start the API service:

```bash
source /root/miniconda3/bin/activate real3dportrait
cd /root/Real3DPortrait
python main_api.py
```

For more detailed debug logs, use:

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --log-level debug
```

Open another terminal and send a request:

```bash
curl -X POST "http://localhost:8000/generate-batch" \
     -H "Content-Type: application/json" \
     -d '{
           "src_image_path": "/root/Real3DPortrait/test/test.png",
           "text_input": "你好。",
           "drv_pose_path": "data/raw/examples/May_5s.mp4",
           "out_mode": "final"
         }'
```

Example response:

```json
{
  "status": "started",
  "task_id": "20260526_162614"
}
```

A successful run should show logs similar to:

```text
HuBERT Pre-loaded
Models and TTS loaded
POST /generate-batch HTTP/1.1" 200 OK
Start rendering 32 frames to .../part_0000.mp4
Successfully saved at .../part_0000.mp4
Batch Task Completed
```

Generated videos are saved under:

```text
infer_out/batch_tasks/{task_id}/
```

---

## 6. API

### POST `/generate-batch`

Start a video generation task.

```bash
curl -X POST "http://localhost:8000/generate-batch" \
     -H "Content-Type: application/json" \
     -d '{
           "src_image_path": "/root/Real3DPortrait/test/test.png",
           "text_input": "你好。",
           "drv_pose_path": "data/raw/examples/May_5s.mp4",
           "out_mode": "final"
         }'
```

Request parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `src_image_path` | string | required | Source portrait image path |
| `text_input` | string | - | Text to synthesize; choose either this or `drv_audio_path` |
| `drv_audio_path` | string | - | Driving audio path, such as `.wav` or `.mp3` |
| `drv_pose_path` | string | `data/raw/examples/May_5s.mp4` | Head pose driving video |
| `bg_image_path` | string | `""` | Background image path; empty value uses the default background logic |
| `blink_mode` | string | `period` | Blink mode, such as `period` or `none` |
| `temperature` | float | `0.2` | Sampling temperature for audio-to-expression generation |
| `mouth_amp` | float | `0.45` | Mouth motion amplitude |
| `out_mode` | string | `final` | `final` for final video only; `concat_debug` for debug visualization |
| `min_face_area_percent` | float | `0.2` | Minimum face area ratio; small faces may be cropped or enlarged |

### GET `/task-status/{task_id}`

Query task status:

```bash
curl "http://localhost:8000/task-status/{task_id}"
```

Possible states:

```text
running / completed / failed
```

### GET `/download/{task_id}/{filename}`

Download a generated video:

```bash
curl -O "http://localhost:8000/download/{task_id}/part_0000.mp4"
```

---

## 7. Troubleshooting

### 1. The API starts, but generation fails with a HuBERT error

Make sure `facebook/hubert-large-ls960-ft` has been downloaded:

```bash
find /root/.cache/huggingface/hub -type d -path "*models--facebook--hubert-large-ls960-ft*" | sort
```

If using a custom location:

```bash
export HUBERT_MODEL_DIR=/path/to/hubert-large-ls960-ft
```

### 2. HuggingFace download times out

Try a mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then rerun the HuBERT download script.

### 3. HuBERT loading requires torch >= 2.6

Do not directly upgrade PyTorch for HuBERT. Use:

```python
HubertModel.from_pretrained(model_name, use_safetensors=True)
```

### 4. Port 8000 is already in use

Check and release the port:

```bash
ss -lntp | grep 8000
fuser -k 8000/tcp
```

Or use another port:

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8001 --log-level debug
```

### 5. No real traceback is shown after startup failure

If the service exits after model loading without a clear traceback, use:

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --log-level debug
```

and keep stderr visible during debugging.

### 6. BFM files are missing

Check:

```bash
ls -lh deep_3drecon/BFM/
```

Make sure large BFM files are placed under:

```text
deep_3drecon/BFM/
```

### 7. `.DS_Store` or `__MACOSX` files

These are usually macOS extraction artifacts. They do not affect runtime and should not be committed to Git.

---

## 8. Directory Structure

```text
real3dportrait-yinfeng/
├── main_api.py
├── tts_engine.py
├── inference/
│   ├── app_real3dportrait.py
│   ├── real3d_infer.py
│   └── infer_utils.py
├── data_gen/
│   └── utils/
│       ├── process_audio/extract_hubert.py
│       └── mp_feature_extractors/selfie_multiclass_256x256.tflite
├── deep_3drecon/
│   └── BFM/
├── modules/
├── checkpoints/
└── infer_out/
    └── batch_tasks/
        └── {task_id}/
            ├── manifest.json
            └── part_XXXX.mp4
```

---

## 9. Reproduction Checklist

Before running the service, check:

1. `checkpoints/` has been prepared;
2. Large BFM files under `deep_3drecon/BFM/` have been prepared;
3. HuBERT `facebook/hubert-large-ls960-ft` has been downloaded;
4. `data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite` exists;
5. `ffmpeg` is available;
6. `infer_out/tts_cache`, `infer_out/batch_tasks` and `infer_out/tmp` exist;
7. the API startup logs include `HuBERT Pre-loaded` and `Models and TTS loaded`;
8. a short text request can generate `part_0000.mp4`;
9. output files appear under `infer_out/batch_tasks/{task_id}/`;
10. large models, generated videos and cache files are not committed to Git.

---

## 10. Acknowledgement

This project is based on [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait). Please refer to the upstream repository for the paper, license and citation information.

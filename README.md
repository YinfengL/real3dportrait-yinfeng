# Real3dportrait-yinfeng

本仓库基于 [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait)（ICLR 2024 Spotlight）进行工程化改造。项目保留原始 Real3D-Portrait 推理链路，并在此基础上新增 FastAPI HTTP 服务、Edge-TTS 文本转语音、长文本分块、异步任务状态管理、静态特征缓存和分块视频生成能力。

项目目标不是重新设计底层 talking-head 生成模型，而是将原始研究型 Demo 改造成一个更适合服务化调用的数字人视频生成模块。典型链路如下：

```text
文本输入 / RAG 文字回答
→ Edge-TTS 生成语音
→ 音频格式统一与 HuBERT 特征提取
→ Real3D-Portrait 推理
→ 数字人口播视频输出
→ 任务状态查询与文件下载
```

本项目已经在 AutoDL Linux + NVIDIA GPU 环境中验证过短文本生成流程。成功运行时应能生成类似：

```text
infer_out/batch_tasks/{task_id}/part_0000.mp4
```

---

## 1. 与原版 Real3D-Portrait 的主要区别

| 维度 | 原版 Real3D-Portrait | 本项目 |
|---|---|---|
| 使用方式 | Gradio WebUI / CLI 推理 | FastAPI HTTP API，同时保留原始推理入口 |
| 输入形式 | 源图像 + 驱动音频 / 驱动视频 | 支持文本输入，由 TTS 自动生成驱动音频；也兼容外部音频输入 |
| 长文本处理 | 原始推理流程未提供服务层分块机制 | 支持自动分句、chunk 级生成和任务级 manifest 记录 |
| 任务执行 | 更偏单次脚本 / Demo 调用 | 后台任务执行，支持 task_id、状态查询和结果下载 |
| 并发控制 | 原始 Demo 未提供 GPU 并发窗口控制 | 使用滑动窗口控制 chunk 推理并发，默认窗口大小为 3 |
| 静态缓存 | 静态视觉预处理与动态音频推理耦合较多 | 对相同人像、姿态、背景的静态特征进行复用 |
| 工程定位 | 论文官方实现 / Demo | 面向服务化调用的数字人视频生成模块 |

---

## 2. 推荐运行环境

建议优先使用 Linux + NVIDIA GPU 服务器，例如 AutoDL 或同等云 GPU 环境。首次复现不建议使用本地 CPU 或 macOS。

原项目推荐环境可参考：

```text
docs/prepare_env/install_guide.md
docs/prepare_env/requirements.txt
```

推荐基础组合：

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

典型安装流程：

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

检查 `ffmpeg`：

```bash
ffmpeg -version
```

> 注意：当前项目曾在 AutoDL / CUDA 12.1 环境中做过服务化适配。如果你的 CUDA 版本不同，可能需要检查 `modules/eg3ds/torch_utils/custom_ops.py` 中自定义 CUDA 算子的 include / library 路径。

---

## 3. 外部模型文件与放置位置

大型模型文件不进入 GitHub 仓库，需要单独下载或从已有运行环境中复制。下面是当前已验证可运行环境中的文件放置方式。

### 3.1 文件放置总表

| 文件类型 | 放置位置 | 是否进入 Git | 说明 |
|---|---|---:|---|
| Real3D-Portrait 预训练权重 | `checkpoints/` | 否 | 包含 `240210_real3dportrait_orig/` 和 `pretrained_ckpts/` |
| audio2secc 权重 | `checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt` | 否 | 必需 |
| secc2plane torso 权重 | `checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt` | 否 | 必需 |
| EG3D baseline 权重 | `checkpoints/pretrained_ckpts/eg3d_baseline_run2/model_ckpt_steps_100000.ckpt` | 否 | 当前验证环境中存在，建议保留 |
| MIT-B0 权重 | `checkpoints/pretrained_ckpts/mit_b0.pth` | 否 | 人像分割 / 编码相关依赖 |
| BFM / 3DMM 大文件 | `deep_3drecon/BFM/` | 否 | 需要单独下载或生成 |
| BFM 小型元数据 | `deep_3drecon/BFM/select_vertex_id.mat` 等 | 是 | 仓库中已保留 |
| HuBERT | `/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/` | 否 | 通过 HuggingFace 或镜像源下载 |
| 自定义 HuBERT 路径 | 任意本地目录，配合 `HUBERT_MODEL_DIR` | 否 | 例如 `pretrained_models/hubert-large-ls960-ft/` |
| MediaPipe 分割模型 | `data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite` | 可进入 Git | 当前环境中已存在 |
| 输出视频 | `infer_out/batch_tasks/{task_id}/` | 否 | 生成产物，不提交 |

---

### 3.2 Real3D-Portrait 预训练权重

从原项目 README 提供的地址下载预训练权重，并解压 / 复制到：

```text
checkpoints/
```

下载地址：

- Google Drive: https://drive.google.com/drive/folders/1MAveJf7RvJ-Opg1f5qhLdoRoC_Gc6nD9?usp=sharing
- 百度网盘: https://pan.baidu.com/s/1Mjmbn0UtA1Zm9owZ7zWNgQ?pwd=6x4f  
  提取码：`6x4f`

当前已验证的目录结构为：

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

启动时可能出现类似 warning：

```text
base.yaml not exist
secc_img2plane_orig.yaml not exist
```

在当前验证环境中，只要上述 `.ckpt` 与 `config.yaml` 文件存在，这类 warning 不一定阻断推理；模型仍可继续加载并完成视频生成。

检查命令：

```bash
cd /root/Real3DPortrait

find checkpoints -maxdepth 4 -type f | sort

ls -lh checkpoints/240210_real3dportrait_orig/audio2secc_vae/
ls -lh checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/
ls -lh checkpoints/pretrained_ckpts/
```

---

### 3.3 BFM / 3DMM 人脸模型

从原项目 README 提供的地址下载 BFM 文件，并放到：

```text
deep_3drecon/BFM/
```

下载地址：

- Google Drive: https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=sharing
- 百度网盘: https://pan.baidu.com/s/1aqv1z_qZ23Vp2VP4uxxblQ?pwd=m9q5  
  提取码：`m9q5`

当前验证环境中的主要文件如下：

```text
deep_3drecon/BFM/
├── 01_MorphableModel.mat              # 需外部下载
├── 3DMM BFM.zip                       # 下载包，可保留作为备份
├── BFM_exp_idx.mat                    # 需外部下载
├── BFM_front_idx.mat                  # 需外部下载
├── BFM_model_front.mat                # 可由转换脚本生成，或由下载包提供
├── Exp_Pca.bin                        # 需外部下载
├── basel_53201.txt                    # 当前环境中存在
├── facemodel_info.mat                 # 需外部下载
├── index_mp468_from_mesh35709.npy     # 仓库已包含
├── index_mp468_from_mesh35709_v1.npy  # 当前环境中存在
├── index_mp468_from_mesh35709_v2.npy  # 当前环境中存在
├── index_mp468_from_mesh35709_v3.npy  # 当前环境中存在
├── index_mp468_from_mesh35709_v3.1.npy# 当前环境中存在
├── select_vertex_id.mat               # 仓库已包含
├── similarity_Lm3D_all.mat            # 仓库已包含
└── std_exp.txt                        # 仓库已包含
```

如果下载包中没有 `BFM_model_front.mat`，可以在首次使用前执行转换脚本：

```bash
cd deep_3drecon/BFM
python -c "from util.load_mats import transferBFM09; transferBFM09('.')"
```

检查命令：

```bash
cd /root/Real3DPortrait
ls -lh deep_3drecon/BFM/
```

---

### 3.4 HuBERT 语音特征模型

本项目在音频驱动推理阶段需要提取 HuBERT 特征。使用的模型是：

```text
facebook/hubert-large-ls960-ft
```

HuBERT 不包含在 GitHub 仓库中。缺少 HuBERT 时，API 可能仍能启动，Edge-TTS 也可能正常加载，但真正生成视频时会在 chunk worker 阶段失败，常见报错为：

```text
Hubert model directory not found
```

#### 推荐下载方式

如果服务器可以访问 HuggingFace 或镜像源，执行：

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

下载完成后，默认缓存路径通常为：

```text
/root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/
```

当前验证环境中，HuBERT 目录结构类似：

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

检查命令：

```bash
find /root/.cache/huggingface/hub -type d -path "*models--facebook--hubert-large-ls960-ft*" | sort

find /root/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots \
  \( -name "model.safetensors" -o -name "pytorch_model.bin" -o -name "config.json" -o -name "preprocessor_config.json" \) \
  2>/dev/null | sort
```

#### 关于 PyTorch 版本

如果加载 `pytorch_model.bin` 时出现需要 `torch >= 2.6` 的报错，不建议为了 HuBERT 直接升级 PyTorch。Real3D-Portrait 依赖的 PyTorch / CUDA / mmcv / pytorch3d 组合较敏感。当前推荐使用 `model.safetensors`：

```python
HubertModel.from_pretrained(model_name, use_safetensors=True)
```

#### 手动指定 HuBERT 路径

当前代码已支持自动从 HuggingFace cache 中查找 snapshot，并优先选择包含 `model.safetensors` 的 snapshot。也可以通过环境变量指定本地 HuBERT 路径：

```bash
export HUBERT_MODEL_DIR=/path/to/hubert-large-ls960-ft
```

例如：

```bash
export HUBERT_MODEL_DIR=/root/Real3DPortrait/pretrained_models/hubert-large-ls960-ft
```

如果把 HuBERT 放到项目目录下的 `pretrained_models/`，不要提交到 Git。

---

### 3.5 MediaPipe 分割模型

当前环境中 MediaPipe segmentation 模型位于：

```text
data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite
```

如果该文件缺失，可以手动下载：

```bash
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

并放置到：

```text
data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite
```

检查命令：

```bash
find . \( -name "*.tflite" -o -iname "*selfie*" -o -iname "*segment*" \) 2>/dev/null | sort
```

---

## 4. 输出目录

任务输出默认保存在：

```text
infer_out/batch_tasks/{task_id}/
```

当前 AutoDL 验证环境中，`infer_out` 是软链接：

```text
infer_out -> /root/autodl-tmp/Real3DPortrait_outputs
```

其实际结构为：

```text
infer_out/
├── batch_tasks/
├── tmp/
└── tts_cache/
```

如果是从 GitHub 新 clone 的仓库，建议先初始化输出目录：

```bash
mkdir -p infer_out/tts_cache infer_out/batch_tasks infer_out/tmp
```

如果你使用 AutoDL 软链接方式，可以先创建真实目标目录：

```bash
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/tts_cache
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/batch_tasks
mkdir -p /root/autodl-tmp/Real3DPortrait_outputs/tmp
```

检查命令：

```bash
ls -lh infer_out
ls -lh infer_out/
ls -lt infer_out/batch_tasks | head
```

---

## 5. 快速启动

启动 API 服务：

```bash
source /root/miniconda3/bin/activate real3dportrait
cd /root/Real3DPortrait
python main_api.py
```

如果需要更详细的调试日志，可以使用：

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --log-level debug
```

另开一个终端请求服务：

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

成功响应示例：

```json
{
  "status": "started",
  "task_id": "20260526_162614"
}
```

成功运行时应看到类似日志：

```text
HuBERT Pre-loaded
Models and TTS loaded
POST /generate-batch HTTP/1.1" 200 OK
Start rendering 32 frames to .../part_0000.mp4
Successfully saved at .../part_0000.mp4
Batch Task Completed
```

输出视频默认保存在：

```text
infer_out/batch_tasks/{task_id}/
```

---

## 6. API 说明

### POST `/generate-batch`

发起视频生成任务。

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

请求参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `src_image_path` | string | 必填 | 源人像图片路径 |
| `text_input` | string | - | 要合成的文本；与 `drv_audio_path` 二选一 |
| `drv_audio_path` | string | - | 驱动音频路径，支持 `.wav` / `.mp3` 等常见格式 |
| `drv_pose_path` | string | `data/raw/examples/May_5s.mp4` | 头部姿态驱动视频 |
| `bg_image_path` | string | `""` | 背景图片路径；为空时使用默认背景处理逻辑 |
| `blink_mode` | string | `period` | 眨眼模式，例如 `period` / `none` |
| `temperature` | float | `0.2` | 音频到表情采样温度；数值越高，随机性越强 |
| `mouth_amp` | float | `0.45` | 嘴部运动幅度；数值越大，张口越明显 |
| `out_mode` | string | `final` | `final` 输出最终视频；`concat_debug` 输出调试视频 |
| `min_face_area_percent` | float | `0.2` | 人脸最小面积比例；人脸过小时会触发裁剪或放大逻辑 |

### GET `/task-status/{task_id}`

查询任务状态：

```bash
curl "http://localhost:8000/task-status/{task_id}"
```

可能状态：

```text
running / completed / failed
```

### GET `/download/{task_id}/{filename}`

下载生成视频：

```bash
curl -O "http://localhost:8000/download/{task_id}/part_0000.mp4"
```

---

## 7. 常见问题

### 1. 服务能启动，但生成任务失败并提示 HuBERT 缺失

确认已经下载：

```text
facebook/hubert-large-ls960-ft
```

并检查：

```bash
find /root/.cache/huggingface/hub -type d -path "*models--facebook--hubert-large-ls960-ft*" | sort
```

如果使用自定义位置，请设置：

```bash
export HUBERT_MODEL_DIR=/path/to/hubert-large-ls960-ft
```

### 2. 下载 HuBERT 时连接 HuggingFace 超时

可以尝试使用镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新执行 HuBERT 下载脚本。

### 3. 加载 HuBERT 时要求升级到 torch >= 2.6

不要直接升级 PyTorch。当前建议使用：

```python
HubertModel.from_pretrained(model_name, use_safetensors=True)
```

### 4. 端口 8000 被占用

检查并释放端口：

```bash
ss -lntp | grep 8000
fuser -k 8000/tcp
```

或者换端口：

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8001 --log-level debug
```

### 5. 启动后看不到真实错误

如果服务加载模型后直接退出但没有 traceback，检查 `main_api.py` 是否重定向了 stderr。调试阶段建议使用：

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --log-level debug
```

并保持错误输出可见。

### 6. BFM 文件缺失

检查：

```bash
ls -lh deep_3drecon/BFM/
```

确保 BFM 大文件已放到：

```text
deep_3drecon/BFM/
```

### 7. `.DS_Store` 或 `__MACOSX` 文件

这些通常是 Mac 解压产生的残留文件，不影响运行，也不需要提交到 Git。

---

## 8. 目录结构

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

## 9. 交接 / 复现检查清单

复现前建议确认：

1. `checkpoints/` 已准备好；
2. `deep_3drecon/BFM/` 中的大型 BFM 文件已准备好；
3. HuBERT `facebook/hubert-large-ls960-ft` 已下载；
4. `data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite` 存在；
5. `ffmpeg` 可用；
6. `infer_out/tts_cache`、`infer_out/batch_tasks`、`infer_out/tmp` 目录存在；
7. API 服务能看到 `HuBERT Pre-loaded` 和 `Models and TTS loaded`；
8. 一个短文本请求能生成 `part_0000.mp4`；
9. 输出文件出现在 `infer_out/batch_tasks/{task_id}/`；
10. 大模型、输出视频、cache 文件不要提交到 Git。

---

## 10. 致谢

本项目基于 [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) 开发。论文、许可证和引用方式请以官方仓库为准。

# real3dportrait-cntech

本仓库基于 [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait)（ICLR 2024 Spotlight）进行工程化改造。项目在保留原始推理链路的基础上，新增 FastAPI HTTP 服务、Edge-TTS 语音合成、长文本分块、异步任务状态管理和分块推理能力，用于内部数字人生成服务与项目交接。

原始 Real3D-Portrait 主要提供 Gradio WebUI 和 CLI 推理方式；本项目更关注服务化调用和工程部署，将“人像图片 + 驱动音频/姿态”的推理流程扩展为“文本输入 → TTS 生成音频 → 数字人视频生成”的服务流程。

---

## 与原版的主要区别

| 维度 | 原版 Real3D-Portrait | 本项目 |
|---|---|---|
| 使用方式 | Gradio WebUI / CLI | FastAPI HTTP API，同时保留原始推理入口 |
| 输入形式 | 源图像 + 驱动音频 / 驱动视频 | 文本输入（TTS 自动合成音频），也兼容音频输入 |
| 长文本处理 | 原始推理流程未提供服务层分块机制 | 自动分句、分块生成、任务级 manifest 记录 |
| 任务执行 | 更偏单次脚本 / Demo 调用 | 后台任务执行，支持状态查询和结果下载 |
| 并发控制 | 原始 Demo 未提供 GPU 并发窗口控制 | 使用滑动窗口并发，默认最多 3 路 chunk 推理 |
| 静态缓存 | 原始流程中静态预处理与动态音频处理耦合较多 | 对相同人像、姿态、背景的静态特征进行复用 |
| 工程定位 | 论文官方实现 / Demo | 内部数字人生成服务与工程交接版本 |

---

## 环境安装

请优先参考原项目安装文档：

```text
docs/prepare_env/install_guide.md
docs/prepare_env/requirements.txt
```

原项目推荐 Python 3.9、PyTorch 2.0.1 和 CUDA 11.7。当前项目曾在 AutoDL 环境中进行服务化适配；如果你的环境是 CUDA 12.1 或其他版本，需要额外检查自定义 CUDA 算子的编译路径。

典型安装流程如下：

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

确认 `ffmpeg` 可用：

```bash
ffmpeg -version
```

---

## 下载模型文件

大型模型文件不进入 Git，需要单独下载或从已有运行环境中复制。当前仓库只保留轻量代码、配置和必要小型元数据文件。

### 1. BFM 人脸模型

从原项目 README 提供的地址下载 BFM 文件，并放到：

```text
deep_3drecon/BFM/
```

下载地址：

- [Google Drive](https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1aqv1z_qZ23Vp2VP4uxxblQ?pwd=m9q5)（提取码：`m9q5`）

建议目录结构如下：

```text
deep_3drecon/BFM/
├── 01_MorphableModel.mat              # 需外部下载
├── BFM_exp_idx.mat                    # 需外部下载
├── BFM_front_idx.mat                  # 需外部下载
├── BFM_model_front.mat                # 由转换脚本生成，或由下载包提供
├── Exp_Pca.bin                        # 需外部下载
├── facemodel_info.mat                 # 需外部下载
├── index_mp468_from_mesh35709.npy     # 仓库已包含
├── select_vertex_id.mat               # 仓库已包含
├── similarity_Lm3D_all.mat            # 仓库已包含
└── std_exp.txt                        # 仓库已包含
```

如果下载包中未直接提供 `BFM_model_front.mat`，首次使用前需要执行一次转换脚本：

```bash
cd deep_3drecon/BFM
python -c "from util.load_mats import transferBFM09; transferBFM09('.')"
```

> 注意：`select_vertex_id.mat` 和 `similarity_Lm3D_all.mat` 是小型必要元数据文件，当前仓库已经包含。如果运行时报这两个文件缺失，请先确认已经拉取最新代码，而不是只检查 BFM 大文件是否下载。

### 2. Real3D-Portrait 预训练权重

从原项目 README 提供的地址下载预训练权重，并解压到：

```text
checkpoints/
```

下载地址：

- [Google Drive](https://drive.google.com/drive/folders/1MAveJf7RvJ-Opg1f5qhLdoRoC_Gc6nD9?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1Mjmbn0UtA1Zm9owZ7zWNgQ?pwd=6x4f)（提取码：`6x4f`）

预期目录结构：

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
    └── mit_b0.pth
```

如果在原 AutoDL 环境中运行，这些文件可能已经存在；如果换新机器，需要从上述链接下载，或从公司共享盘/交接文件中复制。

### 3. HuBERT 语音特征模型

首次运行时，系统会尝试从 HuggingFace 下载：

```text
facebook/hubert-large-ls960-ft
```

如果服务器无法访问 HuggingFace，可以手动下载后放到本地，并修改：

```text
data_gen/utils/process_audio/extract_hubert.py
```

中的 `hubert_model_dir` 路径。

### 4. MediaPipe 分割模型

MediaPipe 分割模型通常会在首次运行时自动下载。如果服务器无外网，可以手动下载：

```bash
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

并根据项目中实际调用路径放置到对应位置。

---

## 启动服务

```bash
cd real3dportrait-cntech
conda activate real3dportrait
export PYTHONPATH=./

python main_api.py
```

服务默认监听：

```text
0.0.0.0:8000
```

启动时会加载 Real3D-Portrait 推理模型和 TTS 模块。看到类似 `Models and TTS loaded` 的日志后，说明服务准备完成。

---

## API 说明

### POST `/generate-batch`：发起生成任务

```bash
curl -X POST http://localhost:8000/generate-batch \
  -H "Content-Type: application/json" \
  -d '{
    "src_image_path": "data/raw/examples/Macron.png",
    "text_input": "你好，我是数字人助手。",
    "drv_pose_path": "data/raw/examples/May_5s.mp4"
  }'
```

返回示例：

```json
{
  "status": "started",
  "task_id": "20240526_143022"
}
```

#### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `src_image_path` | string | 必填 | 源人像图片路径 |
| `text_input` | string | - | 要合成的文本，与 `drv_audio_path` 二选一 |
| `drv_audio_path` | string | - | 驱动音频路径，支持 `.wav` / `.mp3` 等常见格式 |
| `drv_pose_path` | string | `data/raw/examples/May_5s.mp4` | 头部姿态驱动视频 |
| `bg_image_path` | string | `""` | 背景图片路径；为空时使用默认背景处理逻辑 |
| `blink_mode` | string | `period` | 眨眼模式，例如 `period` / `none` |
| `temperature` | float | `0.2` | 音频到表情采样温度；数值越高，随机性越强 |
| `mouth_amp` | float | `0.45` | 嘴部运动幅度；数值越大，张口越明显 |
| `out_mode` | string | `final` | `final` 输出最终视频；`concat_debug` 输出带中间过程的调试视频 |
| `min_face_area_percent` | float | `0.2` | 人脸最小面积比例；人脸过小时会触发裁剪或放大逻辑 |

### GET `/task-status/{task_id}`：查询任务状态

返回示例：

```json
{
  "task_id": "20240526_143022",
  "status": "completed",
  "generated_files": ["part_0000.mp4", "part_0001.mp4"],
  "error": null
}
```

`status` 可能取值：

```text
running / completed / failed
```

### GET `/download/{task_id}/{filename}`：下载生成视频

```bash
curl -O http://localhost:8000/download/20240526_143022/part_0000.mp4
```

### Python 调用示例

```python
import time
import requests

BASE = "http://localhost:8000"

resp = requests.post(f"{BASE}/generate-batch", json={
    "src_image_path": "data/raw/examples/Macron.png",
    "text_input": "欢迎使用数字人服务，今天天气不错。",
    "drv_pose_path": "data/raw/examples/May_5s.mp4",
})

task_id = resp.json()["task_id"]

while True:
    result = requests.get(f"{BASE}/task-status/{task_id}").json()
    if result["status"] in ("completed", "failed"):
        break
    time.sleep(2)

if result["status"] == "completed":
    for filename in result["generated_files"]:
        video = requests.get(f"{BASE}/download/{task_id}/{filename}")
        open(filename, "wb").write(video.content)
else:
    print(result.get("error"))
```

---

## 原始推理脚本测试

在测试 API 前，建议先确认原始推理入口能跑通：

```bash
cd real3dportrait-cntech
conda activate real3dportrait
export PYTHONPATH=./

python inference/real3d_infer.py \
  --src_img data/raw/examples/Macron.png \
  --drv_aud data/raw/examples/Obama_5s.wav \
  --drv_pose data/raw/examples/May_5s.mp4 \
  --bg_img data/raw/examples/bg.png \
  --out_name output.mp4 \
  --out_mode concat_debug
```

如果示例素材不在当前环境中，可以从 upstream 仓库恢复，或替换为自己的图片、音频、姿态视频和背景图。

---

## 目录结构

```text
real3dportrait-cntech/
├── main_api.py                  # FastAPI 服务入口
├── tts_engine.py                # Edge-TTS 封装
├── inference/
│   ├── app_real3dportrait.py    # Inferer 类，模型加载与调用封装
│   ├── real3d_infer.py          # 核心推理逻辑
│   └── infer_utils.py           # 推理辅助函数
├── data_gen/                    # 数据预处理：人脸分割、HuBERT、3DMM 拟合等
├── deep_3drecon/
│   └── BFM/                     # BFM 相关文件；大型文件需手动下载
├── modules/                     # 神经网络模块
├── checkpoints/                 # 模型权重；不进入 Git，需手动下载
└── infer_out/
    └── batch_tasks/
        └── {task_id}/
            ├── manifest.json
            └── part_XXXX.mp4
```

---

## 输出文件

生成任务的输出默认保存在：

```text
infer_out/batch_tasks/{task_id}/
```

通常包含：

```text
manifest.json
part_0000.mp4
part_0001.mp4
...
```

`manifest.json` 用于记录任务状态、输出文件列表和错误信息。

---

## 常见问题

### 启动时报 BFM 相关 `.mat` 或 `.bin` 文件缺失

如果缺少：

```text
01_MorphableModel.mat
BFM_exp_idx.mat
BFM_front_idx.mat
BFM_model_front.mat
Exp_Pca.bin
facemodel_info.mat
```

说明 BFM 大文件尚未准备，请参考“下载模型文件”章节。

如果缺少：

```text
select_vertex_id.mat
similarity_Lm3D_all.mat
```

请先确认仓库已经拉取到最新版本，因为这两个小型元数据文件已保留在 Git 中。

### HuBERT 模型路径报错

检查：

```text
data_gen/utils/process_audio/extract_hubert.py
```

中的 `hubert_model_dir` 配置。如果服务器无法访问 HuggingFace，需要手动下载 `facebook/hubert-large-ls960-ft` 并修改为本地路径。

### CUDA 自定义算子编译失败

当前项目曾在 CUDA 12.1 环境中适配过自定义算子路径。如果你的 CUDA 版本不同，需要检查：

```text
modules/eg3ds/torch_utils/custom_ops.py
```

中的 `extra_include_paths` 和 `extra_ldflags` 是否指向当前机器的 CUDA 路径。

### Edge-TTS 报网络错误

Edge-TTS 依赖微软在线语音服务。请确认服务器能够访问外网，或者替换为内部可用的 TTS 服务。

---

## 交接检查清单

交接前建议确认：

1. GitHub 仓库已同步到最新版本；
2. `checkpoints/` 已通过 Git 之外的方式准备好；
3. `deep_3drecon/BFM/` 中的大型文件已准备好；
4. Conda 环境能正常导入项目模块；
5. `ffmpeg` 可用；
6. 原始 `inference/real3d_infer.py` 能跑通；
7. `main_api.py` 能正常启动；
8. 一个短文本请求能生成至少一个视频片段；
9. 输出文件出现在 `infer_out/batch_tasks/`；
10. 后续开发者已获得私有仓库和外部模型文件的访问权限。

---

## 致谢

本项目基于 [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) 开发。论文、许可证和引用方式请以官方仓库为准。

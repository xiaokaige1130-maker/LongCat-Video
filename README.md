# LongCat-Video

美团开源的基础视频生成模型，支持文生视频、图生视频、视频续写及音频驱动数字人视频生成，参数规模 13.6B。

## 功能特性

- **多任务统一架构**：单一模型同时支持文生视频（Text-to-Video）、图生视频（Image-to-Video）和视频续写（Video-Continuation）
- **长视频生成**：原生预训练支持视频续写，可生成数分钟长视频，无色彩漂移或质量下降
- **高效推理**：采用粗到细生成策略 + 块稀疏注意力（Block Sparse Attention），分钟级生成 720p/30fps 视频
- **多奖励 RLHF 优化**：基于多奖励 GRPO 策略优化，性能对标主流开源及商业方案
- **数字人视频生成（Avatar）**：音频驱动角色动画，支持单人/多人音频输入，兼容动画风格泛化
- **Avatar 1.5 升级**：Whisper-Large 音频编码器实现更精准唇同步，步骤蒸馏加速推理（8 步），支持 INT8 量化

## 技术栈

- **框架**：PyTorch 2.6、Diffusers、Transformers
- **注意力加速**：FlashAttention-2（默认）/ FlashAttention-3 / xFormers
- **音频处理**：Wav2Vec2（v1.0）、Whisper-Large-v3（v1.5）、librosa
- **推理优化**：Context Parallel 多卡并行、torch.compile、INT8 量化
- **交互界面**：Streamlit

## 快速开始

### 安装

```shell
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video
cd LongCat-Video

# 创建 conda 环境
conda create -n longcat-video python=3.10
conda activate longcat-video

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 安装 FlashAttention-2
pip install ninja psutil packaging
pip install flash_attn==2.7.4.post1

# 安装其他依赖
pip install -r requirements.txt

# 安装 Avatar 相关依赖
conda install -c conda-forge librosa ffmpeg
pip install -r requirements_avatar.txt
```

### 下载模型

```shell
pip install "huggingface_hub[cli]"

# 基础视频生成模型
huggingface-cli download meituan-longcat/LongCat-Video --local-dir ./weights/LongCat-Video

# 数字人模型 v1.0
huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar

# 数字人模型 v1.5（推荐）
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir ./weights/LongCat-Video-Avatar-1.5
```

### 运行示例

**文生视频：**
```shell
# 单卡推理
torchrun run_demo_text_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# 多卡推理
torchrun --nproc_per_node=2 run_demo_text_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

**图生视频：**
```shell
torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

**视频续写：**
```shell
torchrun run_demo_video_continuation.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

**长视频生成：**
```shell
torchrun run_demo_long_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

**数字人视频（Avatar 1.5）：**
```shell
# 单人音频驱动
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py \
  --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \
  --stage_1=at2v --input_json=assets/avatar/single_example_1.json \
  --use_distill --model_type avatar-v1.5 --use_int8

# 多人音频驱动
torchrun --nproc_per_node=2 run_demo_avatar_multi_audio_to_video.py \
  --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \
  --input_json=assets/avatar/multi_example_1.json \
  --use_distill --model_type avatar-v1.5 --use_int8
```

**Streamlit 交互界面：**
```shell
streamlit run ./run_streamlit.py --server.fileWatcherType none --server.headless=false
```

## 项目结构

```
LongCat-Video/
├── longcat_video/                  # 核心模块
│   ├── modules/                    # 模型组件（DiT、注意力、LoRA、量化等）
│   │   └── avatar/                 # Avatar 专用模块
│   ├── pipeline_longcat_video.py   # 基础视频生成管线
│   ├── pipeline_longcat_video_avatar.py  # Avatar 视频生成管线
│   ├── audio_process/              # 音频处理（Wav2Vec2 等）
│   ├── block_sparse_attention/     # 块稀疏注意力实现
│   ├── context_parallel/           # 上下文并行推理
│   └── utils/                      # 工具函数（分桶配置、提示词增强）
├── run_demo_text_to_video.py       # 文生视频入口
├── run_demo_image_to_video.py      # 图生视频入口
├── run_demo_video_continuation.py  # 视频续写入口
├── run_demo_long_video.py          # 长视频生成入口
├── run_demo_interactive_video.py   # 交互式视频生成入口
├── run_demo_avatar_single_audio_to_video.py  # 单人数字人入口
├── run_demo_avatar_multi_audio_to_video.py   # 多人数字人入口
├── run_streamlit.py                # Streamlit Web 界面
├── assets/                         # 示例素材和技术报告
├── requirements.txt                # 基础依赖
└── requirements_avatar.txt         # Avatar 额外依赖
```

## 许可证

模型权重及代码采用 MIT 许可证发布。

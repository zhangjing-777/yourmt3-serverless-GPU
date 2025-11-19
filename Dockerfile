# YourMT3 RunPod Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括 git-lfs）
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# 初始化 git-lfs
RUN git lfs install

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip

# 安装 PyTorch (CUDA 11.8，匹配他们的 cu113)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装基础依赖
RUN pip3 install --no-cache-dir boto3 runpod

# 克隆 YourMT3 仓库 (从 Hugging Face) 并强制下载 LFS 文件
RUN git lfs install && \
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/spaces/mimbres/YourMT3 /app/yourmt3 && \
    cd /app/yourmt3 && \
    git lfs pull && \
    ls -lh amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt && \
    echo "Model file size:" && \
    du -h amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt

# 安装 YourMT3 依赖
WORKDIR /app/yourmt3
RUN pip3 install --no-cache-dir \
    python-dotenv \
    yt-dlp \
    mido \
    mir_eval \
    "lightning>=2.2.1" \
    deprecated \
    librosa \
    einops \
    "transformers==4.45.1" \
    "numpy==1.26.4" \
    wandb \
    gradio \
    pretty_midi \
    note-seq

# 尝试安装额外的包（可能失败，但继续）
RUN pip3 install --no-cache-dir \
    https://github.com/coletdjnz/yt-dlp-youtube-oauth2/archive/refs/heads/master.zip \
    || echo "yt-dlp-youtube-oauth2 failed, continuing..."

# 回到工作目录
WORKDIR /app

# 复制处理脚本
COPY src/handler.py /app/handler.py

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python3", "-u", "handler.py"]

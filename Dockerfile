# YourMT3 RunPod Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip

# 安装 PyTorch (CUDA 11.8)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YourMT3 依赖
RUN pip3 install --no-cache-dir \
    transformers \
    librosa \
    soundfile \
    pretty_midi \
    boto3 \
    requests \
    numpy \
    scipy \
    note-seq \
    mir_eval \
    runpod

# 克隆 YourMT3 仓库
RUN git clone https://github.com/mimbres/YourMT3.git /app/yourmt3

# 回到工作目录
WORKDIR /app

# 复制处理脚本
COPY src/handler.py /app/handler.py

# 预下载模型（可选，加快首次启动）
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('mimbres/YourMT3-base', trust_remote_code=True)"

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python3", "-u", "handler.py"]

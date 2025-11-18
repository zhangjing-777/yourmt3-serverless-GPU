import os
import sys
import json
import boto3
import requests
import tempfile
import traceback
import base64
from pathlib import Path
import runpod

# 添加 YourMT3 到 Python 路径
sys.path.append('/app/yourmt3')

from yourmt3 import YourMT3
from yourmt3.config import MT3Config
import pretty_midi

# AWS S3 配置 - 从环境变量读取
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-southeast-1')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'qiupupu')
S3_FOLDER = "yourmt3"

# 初始化 S3 客户端
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 全局模型变量
model = None

def load_model():
    """加载 YourMT3 模型"""
    global model
    if model is None:
        print("Loading YourMT3 model...")
        # 使用 base 模型，单轨钢琴转录
        model = YourMT3.from_pretrained(
            'mimbres/YourMT3-base',
            trust_remote_code=True
        )
        model.eval()
        print("Model loaded successfully!")
    return model

def download_audio(audio_url, temp_dir):
    """从 URL 下载音频文件"""
    print(f"Downloading audio from: {audio_url}")
    response = requests.get(audio_url, timeout=300)
    response.raise_for_status()
    
    # 保存到临时文件
    audio_path = os.path.join(temp_dir, "input_audio.mp3")
    with open(audio_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Audio downloaded to: {audio_path}")
    return audio_path

def upload_to_s3(file_path, s3_key):
    """上传文件到 S3"""
    print(f"Uploading to S3: {s3_key}")
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    
    # 生成可访问的 URL
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    print(f"Upload successful: {url}")
    return url

def transcribe_audio(audio_path, output_dir):
    """使用 YourMT3 转录音频为 MIDI"""
    print("Starting transcription...")
    
    # 加载模型
    mt3_model = load_model()
    
    # 执行转录（单轨钢琴）
    output = mt3_model.transcribe(
        audio_path,
        output_dir=output_dir,
        onset_threshold=0.5,
        offset_threshold=0.5,
        num_instruments=1,  # 单轨
        instrument_names=['piano']  # 钢琴
    )
    
    # 查找生成的 MIDI 文件
    midi_files = list(Path(output_dir).glob("*.mid"))
    if not midi_files:
        raise Exception("No MIDI file generated")
    
    midi_path = str(midi_files[0])
    print(f"Transcription completed: {midi_path}")
    
    return midi_path

def handler(job):
    """RunPod handler 函数"""
    job_input = job['input']
    job_id = job.get('id', 'unknown')
    
    # 支持三种输入方式
    audio_base64 = job_input.get('audio')  # base64 音频数据
    audio_url = job_input.get('audio_url')  # URL
    audio_s3_key = job_input.get('audio_s3_key')  # S3 key
    
    if not any([audio_base64, audio_url, audio_s3_key]):
        return {"error": "需要提供 audio (base64) 或 audio_url 或 audio_s3_key"}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Processing job: {job_id}")
            
            # 1. 获取音频文件
            audio_path = os.path.join(temp_dir, "input_audio.mp3")
            
            if audio_base64:
                # 解码 base64
                print("Decoding base64 audio...")
                audio_data = base64.b64decode(audio_base64)
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                print(f"Saved {len(audio_data)} bytes")
                
            elif audio_url:
                audio_path = download_audio(audio_url, temp_dir)
                
            else:
                print(f"Downloading from S3: {audio_s3_key}")
                s3_client.download_file(S3_BUCKET_NAME, audio_s3_key, audio_path)
            
            # 2. 创建输出目录
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 转录音频
            midi_path = transcribe_audio(audio_path, output_dir)
            
            # 4. 上传到 S3
            output_filename = f"{job_id}.mid"
            s3_key = f"{S3_FOLDER}/{output_filename}"
            midi_url = upload_to_s3(midi_path, s3_key)
            
            # 5. 读取 MIDI 信息（可选）
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            note_count = sum(len(instrument.notes) for instrument in midi_data.instruments)
            duration = midi_data.get_end_time()
            
            # 返回结果
            return {
                "status": "success",
                "midi_url": midi_url,
                "s3_key": s3_key,
                "midi_info": {
                    "note_count": note_count,
                    "duration": round(duration, 2),
                    "instruments": len(midi_data.instruments)
                }
            }
    
    except Exception as e:
        print(f"Error processing job: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    print("Starting YourMT3 RunPod Handler...")
    print(f"S3 Bucket: {S3_BUCKET_NAME}/{S3_FOLDER}")
    
    # 预加载模型
    load_model()
    
    # 启动 RunPod serverless handler
    runpod.serverless.start({"handler": handler})

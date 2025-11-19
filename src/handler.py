import os
import sys
import json
import boto3
import tempfile
import traceback
import base64
import runpod

# 添加 YourMT3 的 amt/src 到路径
sys.path.append(os.path.abspath('/app/yourmt3/amt/src'))

# 导入 YourMT3 的函数
from model_helper import load_model_checkpoint, transcribe
import torchaudio

# AWS S3 配置
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-southeast-1')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'qiupupu')
S3_FOLDER = "yourmt3"

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 全局模型
model = None

def load_yourmt3_model():
    """加载 YourMT3 模型"""
    global model
    if model is None:
        print("Loading YourMT3 model...")
        
        # 使用与 app.py 相同的配置
        model_name = 'YPTF.MoE+Multi (noPS)'
        precision = '16'
        project = '2024'
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
        
        model = load_model_checkpoint(args=args, device="cuda")
        print("Model loaded successfully!")
    return model

def upload_to_s3(file_path, s3_key):
    """上传文件到 S3"""
    print(f"Uploading to S3: {s3_key}")
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    print(f"Upload successful: {url}")
    return url

def handler(job):
    """RunPod handler 函数"""
    job_input = job['input']
    job_id = job.get('id', 'unknown')
    
    # 获取音频
    audio_base64 = job_input.get('audio')
    audio_url = job_input.get('audio_url')
    
    if not audio_base64 and not audio_url:
        return {"error": "需要提供 audio (base64) 或 audio_url"}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Processing job: {job_id}")
            
            # 1. 保存音频文件
            audio_path = os.path.join(temp_dir, "input.mp3")
            
            if audio_base64:
                print("Decoding base64 audio...")
                audio_data = base64.b64decode(audio_base64)
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            else:
                import requests
                print(f"Downloading from URL: {audio_url}")
                response = requests.get(audio_url, timeout=300)
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
            
            # 2. 准备音频信息（模仿 app.py 的 prepare_media）
            info = torchaudio.info(audio_path)
            audio_info = {
                "filepath": audio_path,
                "track_name": f"job_{job_id}",
                "sample_rate": int(info.sample_rate),
                "bits_per_sample": int(info.bits_per_sample) if hasattr(info, 'bits_per_sample') else 16,
                "num_channels": int(info.num_channels),
                "num_frames": int(info.num_frames),
                "duration": int(info.num_frames / info.sample_rate),
                "encoding": str.lower(info.encoding) if hasattr(info, 'encoding') else 'unknown',
            }
            
            # 3. 加载模型并转录
            model = load_yourmt3_model()
            print("Starting transcription...")
            midi_file = transcribe(model, audio_info)
            print(f"Transcription completed: {midi_file}")
            
            # 4. 上传到 S3
            output_filename = f"{job_id}.mid"
            s3_key = f"{S3_FOLDER}/{output_filename}"
            midi_url = upload_to_s3(midi_file, s3_key)
            
            # 5. 返回结果
            return {
                "status": "success",
                "midi_url": midi_url,
                "s3_key": s3_key,
                "job_id": job_id
            }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    print("="*60)
    print("YourMT3 RunPod Handler")
    print("="*60)
    print(f"S3 Bucket: {S3_BUCKET_NAME}/{S3_FOLDER}")
    
    # 预加载模型
    try:
        load_yourmt3_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("Will try to load on first request...")
    
    print("\nStarting RunPod handler...")
    runpod.serverless.start({"handler": handler})

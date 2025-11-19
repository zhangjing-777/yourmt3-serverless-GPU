import os
import sys
import json
import boto3
import tempfile
import traceback
import base64
import runpod

# 添加 YourMT3 的 amt/src 到路径 - 必须在导入前
sys.path.insert(0, '/app/yourmt3/amt/src')
sys.path.insert(0, '/app/yourmt3')

# 先检查路径是否存在
if not os.path.exists('/app/yourmt3/amt/src'):
    print("ERROR: /app/yourmt3/amt/src does not exist!")
    print("Available directories:", os.listdir('/app/yourmt3') if os.path.exists('/app/yourmt3') else "yourmt3 not found")
    sys.exit(1)

# 检查 model_helper.py 是否存在
model_helper_path = '/app/yourmt3/amt/src/model_helper.py'
if not os.path.exists(model_helper_path):
    print(f"ERROR: {model_helper_path} does not exist!")
    print("Files in amt/src:", os.listdir('/app/yourmt3/amt/src') if os.path.exists('/app/yourmt3/amt/src') else "directory not found")
    sys.exit(1)

print(f"Found model_helper.py at: {model_helper_path}")

# 导入 YourMT3 的函数
try:
    from model_helper import load_model_checkpoint, transcribe
    print("Successfully imported model_helper!")
except ImportError as e:
    print(f"Failed to import model_helper: {e}")
    sys.exit(1)

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
    """调试版 handler - 返回目录结构"""
    try:
        info = {}
        
        # 检查目录
        info['yourmt3_exists'] = os.path.exists('/app/yourmt3')
        info['amt_exists'] = os.path.exists('/app/yourmt3/amt')
        info['amt_src_exists'] = os.path.exists('/app/yourmt3/amt/src')
        
        if os.path.exists('/app/yourmt3'):
            info['yourmt3_files'] = os.listdir('/app/yourmt3')[:20]
        
        if os.path.exists('/app/yourmt3/amt'):
            info['amt_files'] = os.listdir('/app/yourmt3/amt')[:20]
        
        if os.path.exists('/app/yourmt3/amt/src'):
            info['amt_src_files'] = os.listdir('/app/yourmt3/amt/src')[:20]
            info['has_model_helper'] = 'model_helper.py' in os.listdir('/app/yourmt3/amt/src')
        
        info['sys_path'] = sys.path[:10]
        
        return {
            "status": "debug",
            "info": info
        }
        
    except Exception as e:
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
    
    # 不预加载，等第一个请求再加载
    print("\nModel will be loaded on first request...")
    
    print("\nStarting RunPod handler...")
    runpod.serverless.start({"handler": handler})

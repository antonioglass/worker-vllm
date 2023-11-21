import os
import argparse
from huggingface_hub import snapshot_download

# Create the parser
parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")

# Add arguments
parser.add_argument('--hugging_face_hub_token', type=str, help='Hugging Face Hub Token')
parser.add_argument('--model_name', type=str, help='Model Name', required=True)
parser.add_argument('--model_revision', type=str, default='main', help='Model Revision')
parser.add_argument('--model_base_path', type=str, default='/workspace/', help='Model Base Path')

# Parse the arguments
args = parser.parse_args()

# Assign values from args
HUGGING_FACE_HUB_TOKEN = args.hugging_face_hub_token
MODEL_NAME = args.model_name
MODEL_REVISION = args.model_revision
MODEL_BASE_PATH = args.model_base_path

# Download the model from hugging face
download_kwargs = {}

if HUGGING_FACE_HUB_TOKEN:
    download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

snapshot_download(
    MODEL_NAME,
    revision=MODEL_REVISION,
    local_dir=f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}",
    local_dir_use_symlinks=False,
    cache_dir="/workspace/",
    **download_kwargs
)

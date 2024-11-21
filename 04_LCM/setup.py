#! python3
"""
Script to download and convert the models to be used in the LCM_Dreamshaper pipeline, then it builds and saves the pipeline. 

Before running this, be sure you have set up a Python virtual env and installed the required libraries as described in the top-level README.md

"""
# Load model directly
import os
from huggingface_hub import snapshot_download
import shutil

model_path = "models"

# Stuff below is from the original example - for converting the older version of the models
import gc
import torch
from pathlib import Path
from diffusers import DiffusionPipeline
import openvino as ov

TEXT_ENCODER_OV_PATH = Path(f"{model_path}/text_encoder.xml")
UNET_OV_PATH = Path(f"{model_path}/unet.xml")
VAE_DECODER_OV_PATH = Path(f"{model_path}/vae_decoder.xml")

skip_safety_checker = False

# Load the LCM model then save it locally
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
save_path = f"{model_path}/LCM_Dreamshaper_v7"
pipe.save_pretrained(save_path)

download_folder = snapshot_download(repo_id="Intel/sd-1.5-lcm-openvino")
cmd = f"copy {download_folder}\*.* {model_path}"
os.system(cmd)

#! python3
"""
Script to download and convert the models to be used for the benchmarks

Before running this, be sure you have set up a Python virtual env and installed the required libraries as described in the top-level README.md

"""

import os, shutil

face_detect_dir = "models/face_detection"
unet_dir = "models/unet-camvid"
mobilenet_dir = "models/mobilenet-v2"

# Download models from the OpenVINO Model Zoo, and convert when necessary
os.system('omz_downloader --name mobilenet-v2-pytorch --output_dir models --cache_dir cache')
os.system('omz_converter --name mobilenet-v2-pytorch --precisions FP16,FP32 --download_dir models --output_dir models')

os.system('omz_downloader --name face-detection-adas-0001 --output_dir models --cache_dir cache')

os.system('omz_downloader --name unet-camvid-onnx-0001 --output_dir models --cache_dir cache')

shutil.move("models/intel/face-detection-adas-0001", face_detect_dir)
shutil.move("models/intel/unet-camvid-onnx-0001", unet_dir)
shutil.move("models/public/mobilenet-v2-pytorch/FP16", f"{mobilenet_dir}/FP16")
shutil.move("models/public/mobilenet-v2-pytorch/FP32", f"{mobilenet_dir}/FP32")
shutil.rmtree("models/intel")
shutil.rmtree("models/public")

# INT8 version of mobilenet had to be quantized manually so it's included in the repo, just move it to the models dir
if not os.path.exists("models/mobilenet-v2-pytorch/INT8"):
    shutil.copytree("mobilenet-v2-int8", f"{mobilenet_dir}/INT8")

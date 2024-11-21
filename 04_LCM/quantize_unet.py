## Quantization

# [NNCF](https://github.com/openvinotoolkit/nncf/) enables post-training quantization by adding quantization layers into a model graph 
# and then using a subset of the training dataset to initialize the parameters of these additional quantization layers. Quantized 
# operations are executed in `INT8` instead of `FP32`/`FP16` making model inference faster.
#
# Since UNet is run iteratively and is the compute bottleneck, we will only quantize UNet. The steps are:
#
# 1. Create a calibration dataset for quantization.
# 2. Run `nncf.quantize()` to obtain quantized model.
# 3. Save the `INT8` model using `openvino.save_model()` function.                          

# This loads a dataset from Hugging Face - "google-research-datasets/conceptual_captions" for calibration. Since quantization
# takes too long for the workshop, we are not storing it locally for use in the workshop.

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import datasets
from pathlib import Path
from tqdm.notebook import tqdm
from transformers import set_seed
import numpy as np
from typing import Any, Dict, List

import openvino as ov
import nncf
from nncf.scopes import IgnoredScope

import OVLatentConsistencyModelPipeline as ov_lcm
from diffusers import DiffusionPipeline

core = ov.Core()

int8_pipe = None
model_path = "model"
num_inference_steps = 4

TEXT_ENCODER_OV_PATH = Path(f"{model_path}/text_encoder.xml")
UNET_OV_PATH = Path(f"{model_path}/unet.xml")
VAE_DECODER_OV_PATH = Path(f"{model_path}/vae_decoder.xml")
UNET_INT8_OV_PATH = Path(f"{model_path}/unet_int8.xml")

class CompiledModelDecorator(ov.CompiledModel):
    def __init__(self, compiled_model, prob: float, data_cache: List[Any] = None):
        super().__init__(compiled_model)
        self.data_cache = data_cache if data_cache else []
        self.prob = np.clip(prob, 0, 1)

    def __call__(self, *args, **kwargs):
        if np.random.rand() >= self.prob:
            self.data_cache.append(*args)
        return super().__call__(*args, **kwargs)

# Prepare calibration dataset
# We use a portion of conceptual_captions(https://huggingface.co/datasets/google-research-datasets/conceptual_captions) dataset from Hugging Face as calibration data.
# To collect intermediate model inputs for calibration we should customize `CompiledModel`.
def collect_calibration_data(lcm_pipeline: ov_lcm.OVLatentConsistencyModelPipeline, subset_size: int) -> List[Dict]:
    original_unet = lcm_pipeline.unet
    lcm_pipeline.unet = CompiledModelDecorator(original_unet, prob=0.3)

    dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True).shuffle(seed=42)
    lcm_pipeline.set_progress_bar_config(disable=True)
    safety_checker = lcm_pipeline.safety_checker
    lcm_pipeline.safety_checker = None

    # Run inference for data collection
    pbar = tqdm(total=subset_size)
    diff = 0
    for batch in dataset:
        prompt = batch["caption"]
        if len(prompt) > tokenizer.model_max_length:
            continue
        _ = lcm_pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil",
            height=512,
            width=512,
        )
        collected_subset_size = len(lcm_pipeline.unet.data_cache)
        if collected_subset_size >= subset_size:
            pbar.update(subset_size - pbar.n)
            break
        pbar.update(collected_subset_size - diff)
        diff = collected_subset_size

    calibration_dataset = lcm_pipeline.unet.data_cache
    lcm_pipeline.set_progress_bar_config(disable=False)
    lcm_pipeline.unet = original_unet
    lcm_pipeline.safety_checker = safety_checker
    return calibration_dataset


device = "CPU"

pipe = DiffusionPipeline.from_pretrained(f"{model_path}/LCM_Dreamshaper_v7")
scheduler = pipe.scheduler
tokenizer = pipe.tokenizer
feature_extractor = pipe.feature_extractor
safety_checker = pipe.safety_checker
text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, device)
unet_fp16 = core.compile_model(UNET_OV_PATH, device)
ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device != "CPU" else {}
vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device, ov_config)
del pipe
ov_pipe = ov_lcm.OVLatentConsistencyModelPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_fp16,
    vae_decoder=vae_decoder,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
    safety_checker=safety_checker,
)

set_seed(1)

if not UNET_INT8_OV_PATH.exists():
    subset_size = 200
    unet_calibration_data = collect_calibration_data(ov_pipe, subset_size=subset_size)

if UNET_INT8_OV_PATH.exists():
    print("Loading quantized model")
    quantized_unet = core.read_model(UNET_INT8_OV_PATH)
else:
    unet = core.read_model(UNET_OV_PATH)
    quantized_unet = nncf.quantize(
        model=unet,
        subset_size=subset_size,
        calibration_dataset=nncf.Dataset(unet_calibration_data),
        model_type=nncf.ModelType.TRANSFORMER,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            disable_bias_correction=True
        )
    )
    ov.save_model(quantized_unet, UNET_INT8_OV_PATH)

unet_optimized = core.compile_model(UNET_INT8_OV_PATH, device)

int8_pipe = ov_lcm.OVLatentConsistencyModelPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc,
    unet=unet_optimized,
    vae_decoder=vae_decoder,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
    safety_checker=safety_checker,
)


# Compare inference times to fp16
import time

validation_size = 10
calibration_dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True)
validation_data = []
for idx, batch in enumerate(calibration_dataset):
    if idx >= validation_size:
        break
    prompt = batch["caption"]
    validation_data.append(prompt)

def calculate_inference_time(pipeline, calibration_dataset):
    inference_time = []
    pipeline.set_progress_bar_config(disable=True)
    for idx, prompt in enumerate(validation_data):
        start = time.perf_counter()
        _ = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=8.0,
            lcm_origin_steps=50,
            output_type="pil",
            height=512,
            width=512,
        )
        end = time.perf_counter()
        delta = end - start
        inference_time.append(delta)
        if idx >= validation_size:
            break
    return np.median(inference_time)

fp_latency = calculate_inference_time(ov_pipe, validation_data)
int8_latency = calculate_inference_time(int8_pipe, validation_data)
print(f"Performance speed up: {fp_latency / int8_latency:.3f}")

fp16_ir_model_size = UNET_OV_PATH.with_suffix(".bin").stat().st_size / 1024
quantized_model_size = UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 1024

print(f"FP16 model size: {fp16_ir_model_size:.2f} KB")
print(f"INT8 model size: {quantized_model_size:.2f} KB")
print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")

exit()








quantized_unet = run_quantization(ov_pipe, UNET_OV_PATH, UNET_INT8_OV_PATH)

fp16_ir_model_size = UNET_OV_PATH.with_suffix(".bin").stat().st_size / 1024
quantized_model_size = UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 1024

print(f"FP16 model size: {fp16_ir_model_size:.2f} KB")
print(f"INT8 model size: {quantized_model_size:.2f} KB")
print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")
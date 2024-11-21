#! python3
"""
Script to download and convert the models to be used in the MusicGen pipeline. 

Before running this, be sure you have set up a Python virtual env and installed the required libraries as described in the top-level README.md
 
To use static-shaped inputs for the text encoder, be sure to use OpenVINO 2024.4 or newer
""" 
import os

from transformers import AutoProcessor, MusicgenForConditionalGeneration, T5Tokenizer, EncodecFeatureExtractor

import sys, gc
from pathlib import Path
from packaging.version import parse
import importlib.metadata as importlib_metadata

model_path = "./models"
loading_kwargs = {}

if parse(importlib_metadata.version("transformers")) >= parse("4.40.0"):
    loading_kwargs["attn_implementation"] = "eager"

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torchscript=True, return_dict=False, **loading_kwargs)

# tokenizer = T5Tokenizer.from_pretrained("facebook/musicgen-small", **loading_kwargs)
# feature_extractor = EncodecFeatureExtractor.from_pretrained("facebook/musicgen-small", **loading_kwargs)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

musicgen_path = f"{model_path}/musicgen-small"
model.save_pretrained(musicgen_path)

processor_path = f"{model_path}/musicgen-small-processor"
processor.save_pretrained(processor_path)
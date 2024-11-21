import torch
import openvino as ov
import numpy as np
import random

import gradio as gr
from functools import partial

core = ov.Core()

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    MAX_SEED = np.iinfo(np.int32).max
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def build_gr_blocks(call_fn):

    device_options = core.available_devices + ["AUTO"]
    examples = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour",
        "style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
    result = gr.Image(label="Result", type="pil", streaming=True)
        
    with gr.Blocks() as demo:
        with gr.Accordion(label="Expand to choose from example prompts", open=False):
            gr.Examples(examples=examples, inputs=prompt, outputs=result, cache_examples=False)
        with gr.Row():
                prompt.render()
        with gr.Row():
            with gr.Column(scale=1):
                    text_enc_device = gr.Dropdown(device_options, label="Text Encoder Device", value=device_options[-1])
                    # "AUTO" is not working with this version of UNet so set this up differently
                    unet_device = gr.Dropdown(core.available_devices, label="UNet Device", value="GPU")
                    vae_device = gr.Dropdown(device_options, label="VAE Device", value=device_options[-1])
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4,
                    )
                    msgs = gr.Textbox(label="Output Messages")
            with gr.Column(scale=4):
                    result.render()
                    run_button = gr.Button("Run", variant="primary")             
        
        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=call_fn,
            inputs=[
                text_enc_device,
                unet_device,
                vae_device,
                prompt,
                num_inference_steps,
            ],
            outputs=[result, msgs],
        )

    return demo


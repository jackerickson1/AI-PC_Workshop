import torch
import openvino as ov
from collections import namedtuple
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import gradio as gr

core = ov.Core()

def build_gr_blocks(call_fn):
    examples = [
        ["80s pop track with bassy drums and synth"],
        ["Earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves"],
        ["90s rock song with loud guitars and heavy drums"],
        ["Heartful EDM with beautiful synths and chords"],
    ]
    prompt = gr.Textbox(label="Text Prompt")
    music = gr.Audio(label="Generate Music")
    
    with gr.Blocks(fill_width=True) as demo:
        gr.Markdown("### Generate Music Locally on an AI PC")
        with gr.Accordion(label="Expand to choose from example prompts", open=False):
            gr.Examples(examples=examples, inputs=prompt, outputs=music, cache_examples=False)
        with gr.Row():
            with gr.Column(scale=2, min_width=200):
                prompt.render()
                output_length = gr.Dropdown([5, 10, 20], label="Length of output, in seconds", value=5)
                submit = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1, min_width=100):
                music.render()
            submit.click(call_fn, inputs=[prompt, output_length], outputs=[music])

    return demo

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder_ir, config, device):
        super().__init__()
        self.encoder = core.compile_model(encoder_ir, device)
        self.config = config

    def forward(self, input_ids, **kwargs):
        last_hidden_state = self.encoder(input_ids)[self.encoder.outputs[0]]
        last_hidden_state = torch.tensor(last_hidden_state)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)

class AudioDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder_ir, config, device):
        super().__init__()
        print(f"Loading {decoder_ir} on {device}")
        self.decoder = core.compile_model(decoder_ir, device)
        self.config = config
        self.output_type = namedtuple("AudioDecoderOutput", ["audio_values"])

    def decode(self, output_ids, audio_scales):
        output = self.decoder(output_ids)[self.decoder.outputs[0]]
        return self.output_type(audio_values=torch.tensor(output))
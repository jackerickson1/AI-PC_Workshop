# AI PC Hands-on Workshop Modules

This repo contains the modules for the AI PC hands-on workshops from Q4 2024. There are 5 modules:

1. Introduction: demonstrates an application that applies a variety of style transfer filters to your live camera input, and the ability to swap between different inference devices.
2. Benchmark App: shows how to use OpenVINO utilities get more information on your AI PC processor, and shows how to use the OpenVINO benchmark_app to compare latency or throughput for the CPU, GPU, and NPU processor cores for a variety of models and precisions.
3. Chatbot with RAG: build a LangChain-based chatbot pipeline, ask it some questions about the newest Intel Core Ultra processor (code-named Lunar Lake). Then add retrieval-augmented generation (RAG) to ground the LLM's knowledge in actual documents about this processor and see how the answers change. This demonstrates the security/privacy value of local inference - everything stays on your PC.
4. Latent Consistency Model: Text-to-image generation using [latent consistency](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm). You can experiment with different components of the pipeline running on different devices.
5. Music Generation: generate music using the Meta MusicGen pipeline, which consists of 3 models. These models all use dynamically-shaped inputs - running them on the NPU requires statically-shaped inputs, so part of this module discusses strategies for converting to static shapes, and two of the models are converted.

### Python package installation

The modules use a variety of packages. You can either install the packages for a specific module, or install all the modules' packages at once. It's best to use a virtual environment for this. Here are the two methods:

Option a: install all packages for all the AI PC workshop modules. _From the top directory_:
```
python -m venv env
env\Scripts\activate
pip install -q -r requirements_all.txt
```   
Option b: only install packages for a given module. _In a given module's directory_:
```
python -m venv env
env\Scripts\activate
pip install -q -r requirements.txt`
```
### Model downloads

This workshop was designed to be run locally on laptops that might not have internet access, so each module has a `setup.py` Python script that downloads models ahead of time and saves them locally. So in each module, be sure to run this script before you try to use the notebooks.
```
python setup.py
```

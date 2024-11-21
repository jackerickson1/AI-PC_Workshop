#! python3
"""
Script to download and convert the models for the RAG chain, and download some example documents for the knowledge database.

Before running this, be sure you have set up a Python virtual env and installed the required libraries as described in the top-level README.md

"""

import os, shutil, requests, io
from pathlib import Path
import openvino as ov
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher

# Download/convert the LLM. The workshop only uses the INT4 model, the other commands are here in case you prefer those precisions
# os.system('optimum-cli export openvino --model Intel/neural-chat-7b-v3-3 --task text-generation-with-past --weight-format fp16 models/neural-chat-7b-v3-3/FP16')
# os.system('optimum-cli export openvino --model Intel/neural-chat-7b-v3-3 --task text-generation-with-past --weight-format int8 models/neural-chat-7b-v3-3/INT8')
os.system('optimum-cli export openvino --model Intel/neural-chat-7b-v3-3-int4-inc models/neural-chat-7b-v3-3/INT4')

# Embeddings model
os.system('optimum-cli export openvino --model BAAI/bge-small-en-v1.5 --task feature-extraction models/bge-small-en-v1.5')
# Rerank model
os.system('optimum-cli export openvino --model BAAI/bge-reranker-v2-m3 --task text-classification models/bge-reranker-v2-m3')

# Create/optimize an NPU version of the embedding model
embedding_model_dir = "./models/bge-small-en-v1.5"
embedding_model_name = Path(embedding_model_dir) / "openvino_model.xml"
npu_embedding_dir = embedding_model_dir + "-npu"
npu_embedding_path = Path(npu_embedding_dir) / "openvino_model.xml"
packed_layername_tensor_dict_list = [{"name": "aten::mul/Multiply"}]

if not Path(npu_embedding_dir).exists():
    shutil.copytree(embedding_model_dir, npu_embedding_dir)
core = ov.Core()
ov_model = core.read_model(embedding_model_name)

# The following comes from https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py
class ReplaceTensor(MatcherPass):
    def __init__(self, packed_layername_tensor_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Multiply")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            for y in packed_layername_tensor_dict_list:
                root_name = root.get_friendly_name()
                if root_name.find(y["name"]) != -1:
                    max_fp16 = np.array([[[[-np.finfo(np.float16).max]]]]).astype(np.float32)
                    new_tenser = ops.constant(max_fp16, Type.f32, name="Constant_4431")
                    root.set_arguments([root.input_value(0).node, new_tenser])
                    packed_layername_tensor_dict_list.remove(y)

            return True

        self.register_matcher(Matcher(param, "ReplaceTensor"), callback)


manager = Manager()
manager.register_pass(ReplaceTensor(packed_layername_tensor_dict_list))
manager.run_passes(ov_model)
ov.save_model(ov_model, npu_embedding_path, compress_to_fp16=False)
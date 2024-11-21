import gradio as gr

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

RAG_PROMPT_TEMPLATE = """<|system|> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<|user|>
Question: {input} 
Context: {context} 
Answer: 
<|assistant|>"""

from langchain_community.embeddings import OpenVINOBgeEmbeddings
def load_embeddings_model(embedding_device):

    embedding_model_dir = "./models/bge-small-en-v1.5"

    USING_NPU = embedding_device == "NPU"

    npu_embedding_dir = embedding_model_dir + "-npu"
    
    embedding_model_name = npu_embedding_dir if USING_NPU else embedding_model_dir
    batch_size = 1 if USING_NPU else 4
    embedding_model_kwargs = {"device": embedding_device, "compile": False}
    encode_kwargs = {
        "mean_pooling": False,
        "normalize_embeddings": True,
        "batch_size": batch_size,
    }

    embedding = OpenVINOBgeEmbeddings(
        model_name_or_path=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    if USING_NPU:
        embedding.ov_model.reshape(1, 512)
    embedding.ov_model.compile()
    return embedding

from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
def load_rerank_model(device):

    reranker = OpenVINOReranker(
        model_name_or_path="./models/bge-reranker-v2-m3",
        model_kwargs={"device": device},
        top_n=2,
    )
    return reranker

# Build the Gradio blocks interface to run the demo
def build_gr_blocks(call_fn, use_rag):

    examples = [
        "What is Lunar Lake",
        "What are the benefits of Lunar Lake?",
        "How many total TOPS is Lunar Lake?",
        "What types of compute cores are in Lunar Lake?",
    ]
    query = gr.Textbox(label="Question", placeholder="Ask a question")
    response = gr.Textbox(label="Response")
        
    with gr.Blocks(fill_width=True) as demo:
        if use_rag:
            gr.Markdown("# Question Answering with Retrieval")
        else:
            gr.Markdown("# Question Answering without Retrieval")
        with gr.Accordion(label="Expand to choose from example prompts", open=False):
            gr.Examples(examples=examples, inputs=query, outputs=response, cache_examples=False)
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                query.render()
                # device = gr.Dropdown(["CPU", "NPU"], label="Inference device", value="CPU")
                use_rag_inp = gr.Checkbox(label="Use RAG", value=use_rag, visible=False)                
                submit = gr.Button("Submit", variant="primary")
            with gr.Column(scale=2, min_width=300):
                response.render()
            submit.click(call_fn, inputs=[query, use_rag_inp], outputs=[response])
    return demo

def partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text
    

# Parse the prompt from the Langchain output to return only the answer
def parse_output(response):
    # Find the position of "<|assistant|>" in the response and return the text after it.
    answer_start = response.find("<|assistant|>")
    if answer_start != -1:
        answer = response[answer_start + len("<|assistant|>"):].strip()
        return answer
    return response.strip()



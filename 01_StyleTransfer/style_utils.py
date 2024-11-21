import collections
import time

import cv2
import numpy as np
from pathlib import Path
from IPython.display import display, clear_output, Image
import openvino as ov
import threading
import gradio as gr
from importlib.metadata import version

base_model_dir = "models"

# Global variables to control the video stream and filter
vid_in = None
use_popup = True
stop_processing = False
USE_WEBCAM_state = True
H=0
W=0
output_layer=0
thread_active=False
no_filter=False

core=ov.Core()

# Define the style options and their scatter plot markers
style_options = ["mosaic", "rain-princess", "candy", "udnie", "pointilism", "No filter"]

device_options = core.available_devices

def build_gr_interface():
    global compiled_model_dict
    
    compiled_model_dict = load_models(style_options, device_options)

    inputs = [
        gr.Dropdown(style_options, label="Choose a style", value=None, interactive=True),
        gr.Dropdown(device_options, label="Choose a device", value=device_options[0], interactive=True),
        gr.Checkbox(label="Use webcam?", value=True),
    ]

    # Ugh
    if version('gradio')[0] != '5':
        gr_if = gr.Interface(
            fn=controller, 
            inputs=inputs, 
            clear_btn=gr.Button("Clear inputs and stop", variant="stop"),
            outputs=gr.Text(label="Status"),
            live=True,
            allow_flagging="never",
        )
    else:
        gr_if = gr.Interface(
            fn=controller, 
            inputs=inputs, 
            clear_btn=gr.Button("Clear inputs and stop", variant="stop"),
            outputs=gr.Text(label="Status"),
            live=True,
            flagging_mode="never", 
        )

    return gr_if

# This is the function that gets called from the Gradio interface, so it receives all the inputs from the various dropdowns, etc.
# Once it sets things up, it spawns a thread for the main processing loop to free-run, then just returns a status message to 
# the Gradio interface so it can continue waiting for new inputs. 
# There are a lot of global variables here to make changes to that free-running thread - this is the "hot swap" capability.

def controller(model_name, device, USE_WEBCAM):
    global compiled_model, compiled_model_dict, USE_WEBCAM_state, vid_in, H, W, thread_active, no_filter
    
    # Typically, the primary webcam is set with source=0. If you have multiple webcams, each one will be assigned a consecutive 
    #     number starting at 0. 
    cam_id = 0
    # Set flip=True when using a front-facing camera. 
    # Some web browsers, especially Mozilla Firefox, may cause flickering. If you experience flickering, set use_popup=True. 
    video_file = "Coco%20Walking%20in%20Berkeley.mp4"
    flip = True if USE_WEBCAM else False
    source = cam_id if USE_WEBCAM else video_file
    
    if model_name == None or device == None:
        cleanup()
        status_msg = "Stopped"
        return status_msg

    if model_name == "No filter":
        no_filter = True
        # Just manually set H,W
        H = 224
        W = 224
    else:
        no_filter = False
        # Get the compiled model from the dict
        key_name = f"{model_name}_{device}"
        compiled_model = compiled_model_dict.get(key_name)
        # Get the input and output nodes.
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        # Get the model's input layer size.
        N, C, H, W = list(input_layer.shape)
    
    # Initialize vid_in, or restart vid_in if USE_WEBCAM setting changed
    if vid_in == None or USE_WEBCAM_state != USE_WEBCAM:
        vid_in = cv2.VideoCapture(source)
        if not vid_in.isOpened():
            print("Error: Could not open webcam.")
        else:        
            USE_WEBCAM_state = USE_WEBCAM
            if not thread_active:
                vid_thread = threading.Thread(target=run_style_transfer)
                vid_thread.start()
                thread_active = True

    status_msg = f"Running {model_name} on {device}"
    return status_msg           
    

# The style transfer function can be run in different operating modes, either using a webcam or a video file.
# Playback happens in a popup window. Set use_popup=True in the globals. 
def run_style_transfer():

    global compiled_model, vid_in, use_popup, stop_processing, output_layer, H, W, USE_WEBCAM_state, no_filter
    
    if use_popup:
        title = "Press ESC to Exit"
        cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    processing_times = collections.deque()

    while not stop_processing:
        
        # Grab the frame.
        ret, frame = vid_in.read()
        if not ret:
            print("Source ended")
            return
        if USE_WEBCAM_state == True:
            frame = cv2.flip(frame, 1)
        # If the frame is larger than full HD, reduce size to improve the performance.
        scale = 720 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        # Preprocess the input image.
        image = preprocess_images(frame, H, W)

        # Measure processing time for the input image.
        start_time = time.time()
            
        # Perform the inference step.
        if no_filter:
            stylized_image = image
        else:
            stylized_image = compiled_model([image])[output_layer]
        stop_time = time.time()

        # Postprocessing for stylized image.
        result_image = convert_result_to_image(frame, stylized_image)
        processing_times.append(stop_time - start_time)
            
        # Use processing times from last 200 frames.
        if len(processing_times) > 200:
            processing_times.popleft()
        processing_time_det = np.mean(processing_times) * 1000

        # Visualize the results.
        f_height, f_width = frame.shape[:2]
        fps = 1000 / processing_time_det
        cv2.putText(
            result_image,
            text=f"Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Use this workaround if there is flickering.
        if use_popup:
            cv2.imshow(title, result_image)
            key = cv2.waitKey(1)
            # <Esc> = 27
            if key == 27:
                cleanup()
        else:
            # Various attempts to get this to play in the notebook
            # Encode numpy array to jpg.
            _, encoded_img = cv2.imencode(".jpg", result_image, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
            # cv2.imwrite(temp_file.name, result_image)
            # Create an IPython image.
            i = Image(data=encoded_img)
            # Display the image in this notebook.
            clear_output(wait=True)
            display(i)
            

# Preprocess the input image.
# 1. Preprocess a frame to convert from `unit8` to `float32`.
# 2. Transpose the array to match with the network input size
def preprocess_images(frame, H, W):
    """
    Preprocess input image to align with network size

    Parameters:
        :param frame:  input frame
        :param H:  height of the frame to style transfer model
        :param W:  width of the frame to style transfer model
        :returns: resized and transposed frame
    """
    image = np.array(frame).astype("float32")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(src=image, dsize=(H, W), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    return image
    
# Postprocess the result
# The converted IR model outputs a NumPy `float32` array of the shape [(1, 3, 224, 224)]
def convert_result_to_image(frame, stylized_image) -> np.ndarray:
    """
    Postprocess stylized image for visualization

    Parameters:
        :param frame:  input frame
        :param stylized_image:  stylized image with specific style applied
        :returns: resized stylized image for visualization
    """
    h, w = frame.shape[:2]
    stylized_image = stylized_image.squeeze().transpose(1, 2, 0)
    stylized_image = cv2.resize(src=stylized_image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
    return stylized_image
    
# Loop through all the models and loads all of them on all the available devices. This is not a typical use case! 
# However this enables us to hot-swap the video between different models and different devices. 
# And the AI PC is more than capable of handling this.
def load_models(model_list, device_list):

    compiled_model_dict = {}

    for model_name in model_list:
        if model_name != "No filter":
            for device in device_list:

                # Converted IR model path
                ir_path = Path(f"{base_model_dir}/{model_name}-9.xml")
                key_name = f"{model_name}_{device}"

                # Read the network and corresponding weights from IR Model.
                model = core.read_model(model=ir_path)
                compiled_model = core.compile_model(model=model, device_name=device)
                compiled_model_dict.update({key_name : compiled_model})

    return compiled_model_dict

# Shut down video inputs, outputs, and windows.
def cleanup():
    global vid_in, use_popup, stop_processing, thread_active

    # Stop the processing loop
    stop_processing = True
    thread_active = False
    
    if vid_in is not None:
        # Stop capturing.
        vid_in.release()
        vid_in = None
    if use_popup:
        cv2.destroyAllWindows()
    
    # Re-enable for future restarts
    stop_processing = False
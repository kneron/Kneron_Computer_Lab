import argparse
import os
import ctypes
import sys
import cv2
import time
from common import constants
from python_wrapper import kdp_wrapper
from python_wrapper import kdp_examples
from kdp_host_api import (kdp_add_dev, kdp_init_log, kdp_lib_de_init, kdp_lib_init, kdp_lib_start)

# Read class labels
labels_file = 'python_wrapper/coco.names'
labels_path = os.path.join(os.getcwd(), labels_file)
labels = open(labels_path).read().strip().split("\n")

def handle_result(inf_res, r_size, frames):
    """Handle the detected results returned from the model.

    Arguments:
        inf_res: Inference result data.
        r_size: Inference data size.
        frames: List of frames captured by the video capture instance.
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if r_size >= 4:
        header_result = ctypes.cast(
            ctypes.byref(inf_res), ctypes.POINTER(constants.ObjectDetectionRes)).contents
        box_result = ctypes.cast(
            ctypes.byref(header_result.boxes),
            ctypes.POINTER(constants.BoundingBox * header_result.box_count)).contents
        for box in box_result:
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
            class_number = int(box.class_num)
            class_labels = labels[class_number]
            cv2.putText(frames[0], class_labels,(x1,y1), font, 1, (255, 255, 255), 2, cv2.LINE_AA)   

        cv2.imshow('detection', frames[0])
        del frames[0]
        key = cv2.waitKey(1)

        if key == ord('q'):
            sys.exit()
    return 0
    
# Define KL520 parameters
KDP_UART_DEV    = 0
KDP_USB_DEV     = 1
image_source_w	= 640
image_source_h	= 480
image_size      = image_source_w * image_source_h * 2
loop_count      = 1000
app_id          = constants.APP_TINY_YOLO3
user_id         = 0
frames          = []

# Read input arguments
parser = argparse.ArgumentParser(description="Run yolo v3 object detection in serial, pipeline, parallel modes")
parser.add_argument('-t', '--task_name', help=("serial\npipeline\nparallel"))
args = parser.parse_args()

# Initialize Kneron USB device
kdp_init_log("/tmp/", "mzt.log")

print("Initialize kdp host lib  ....\n")
if (kdp_lib_init() < 0):
    print("Initialize kdp host lib failure\n")

print("Add kdp device ....")
dev_idx = kdp_add_dev(KDP_USB_DEV, "")
if (dev_idx < 0):
    print("Add kdp device failure\n")

print("Start kdp host lib ....\n")
if (kdp_lib_start() < 0):
    print("Start kdp host lib failure")

print("Start kdp task: ", args.task_name)

# Setup video capture device.
capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
if capture is None:
    exit()

if (args.task_name == "serial"):

    # Start ISI mode.
    if kdp_wrapper.start_isi(dev_idx, app_id, image_source_w, image_source_h):
        exit()

    img_id_tx   = 0
    start_time = time.time()
    while (img_id_tx != loop_count):
        kdp_wrapper.sync_inference(dev_idx, app_id, image_size, capture, img_id_tx, frames, handle_result)
        img_id_tx += 1

elif (args.task_name == "pipeline"):

    # Start ISI mode.
    if kdp_wrapper.start_isi(dev_idx, app_id, image_source_w, image_source_h):
        exit()

    start_time = time.time()
    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.fill_buffer(dev_idx, capture, image_size, frames)
    if ret:
        exit()

    kdp_wrapper.pipeline_inference(
        dev_idx, app_id, loop_count - buffer_depth, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result)

elif (args.task_name == "parallel"):

    # Start ISI mode.
    if kdp_wrapper.start_isi_parallel(dev_idx, app_id, image_source_w, image_source_h):
        exit()

    start_time = time.time()
    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.fill_buffer(dev_idx, capture, image_size, frames)
    if ret:
        exit()

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    kdp_wrapper.pipeline_inference(
        dev_idx, app_id, loop_count - buffer_depth, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result)

end_time = time.time()
diff = end_time - start_time 
estimate_runtime = float(diff/loop_count)
fps = float(1/estimate_runtime)    
print(args.task_name, "inference runtime : ", estimate_runtime)
print("Average FPS is ", fps)

# Exit Kneron USB device
print("Exit kdp host lib ....\n")
kdp_lib_de_init()


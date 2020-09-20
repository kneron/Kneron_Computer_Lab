import argparse
import os
import cv2
import ctypes
from common import constants
from python_wrapper import kdp_wrapper
from python_wrapper import kdp_examples
from kdp_host_api import (kdp_add_dev, kdp_init_log, kdp_lib_de_init, kdp_lib_init, kdp_lib_start)

# Define KL520 parameters
KDP_UART_DEV    = 0
KDP_USB_DEV     = 1
IMG_SRC_WIDTH	= 640
IMG_SRC_HEIGHT	= 480
ISI_YOLO_ID     = constants.APP_TINY_YOLO3
image_size      = IMG_SRC_WIDTH * IMG_SRC_HEIGHT * 2
user_id         = 0

def detect_image(dev_idx, user_id):

    # Initialize image capture parameters
    frames      = []
    img_id_tx   = 0

    # Setup image path
    data_path = os.path.join(os.getcwd(), 'images')

    # Read input image
    image_name = input('Input image file: ')
    image_path = os.path.join(data_path, image_name)
    image_flag = os.path.isfile(image_path)

    # Start ISI mode
    if (kdp_wrapper.start_isi(dev_idx, ISI_YOLO_ID, IMG_SRC_WIDTH, IMG_SRC_HEIGHT)):
        return -1
    
    # Perform image inference
    while image_flag:
        image = cv2.imread(image_path)
        kdp_examples.image_inference(dev_idx, ISI_YOLO_ID, image_size, image, img_id_tx, frames)
        img_id_tx += 1
        image_name = input('Input image file: ')
        image_path = os.path.join(data_path, image_name)
        image_flag = os.path.isfile(image_path)

    cv2.destroyAllWindows()

def detect_camera(dev_idx, user_id):

    # Initialize camera capture parameters
    frames      = []
    img_id_tx   = 0

    # Setup webcam capture
    capture     = kdp_wrapper.setup_capture(0, IMG_SRC_WIDTH, IMG_SRC_HEIGHT)
    if capture is None:
        print("Can't open webcam")
        return -1

    # Start ISI mode
    if (kdp_wrapper.start_isi(dev_idx, ISI_YOLO_ID, IMG_SRC_WIDTH, IMG_SRC_HEIGHT)):
        return -1

    # Perform video inference
    while True:
        kdp_examples.camera_inference(dev_idx, ISI_YOLO_ID, image_size, capture, img_id_tx, frames)
        img_id_tx += 1

    capture.release()
    cv2.destroyAllWindows()

# Read input arguments
parser = argparse.ArgumentParser(description="Run yolo v3 object detection")
parser.add_argument('-t', '--task_name', help=("image; camera"))
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

if (args.task_name == "image"):
    detect_image(dev_idx, user_id)
elif (args.task_name == "camera"):
    detect_camera(dev_idx, user_id)

# Exit Kneron USB device
print("Exit kdp host lib ....\n")
kdp_lib_de_init()

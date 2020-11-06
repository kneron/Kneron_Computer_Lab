"""
Name	: kdp_examples.py
Author	: Oscar Law

	Develop multiple routines to support Kneron NPU operation

Routines:
Jul 17, 2020. capture_image(image, frames)
Jul 17, 2020. capture_tello(frame, frames)
Jul 17, 2020. display_image(inf_res, r_size, frames)
Jul 17, 2020. display_tello(inf_res, r_size, frames)
Jul 17, 2020. camera_inference(device_index, app_id, input_size, capture, img_id_tx, frames)
Jul 17, 2020. image_inference(device_index, app_id, input_size, image_path, img_id_tx, frames)
Jul 17, 2020. tello_inference(device_index, app_id, input_size, frame_read, img_id_tx, frames)

History:
Aug 04, 2020. O. Law
- modified from host_lib v0.30
Aug 06, 2020. O. Law
- merge capture_image and capture_tello together
- merge display_image and display_tello together

Copyright (c) 2020 Oscar Law
All Rights Reserved.
"""

from __future__ import absolute_import
import ctypes
import math
import sys
from time import sleep
import cv2
import os
import numpy as np
from common import constants
from python_wrapper import kdp_wrapper
import kdp_host_api as api

# Read class labels
labels_file = 'python_wrapper/coco.names'
labels_path = os.path.join(os.getcwd(), labels_file)
labels = open(labels_path).read().strip().split("\n")

# Define image parameters
IMG_SOURCE_W = 640
IMG_SOURCE_H = 480
DME_IMG_SIZE = IMG_SOURCE_W * IMG_SOURCE_H * 2

def capture_image(image, frames):

    frame = image
    frame = cv2.resize(frame, (IMG_SOURCE_W, IMG_SOURCE_H), interpolation=cv2.INTER_CUBIC)
    frames.append(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(DME_IMG_SIZE)
    buf_len = DME_IMG_SIZE
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p

def display_image(inf_res, r_size, frames):
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
            cv2.putText(frames[0],class_labels,(x1,y1), font, 1, (255, 255, 255), 2, cv2.LINE_AA)         
            
        cv2.imshow('detection', frames[0])
        del frames[0]

    return 0

def camera_inference(device_index, app_id, input_size, capture,
                  img_id_tx, frames):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        input_size: Size of input image.
        ret_size: Return size.
        capture: Active cv2 video capture instance.
        img_id_tx: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.

    """
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()

    data_p = kdp_wrapper.isi_capture_frame(capture, frames)

    ret, _, img_left = kdp_wrapper.isi_inference(
        device_index, data_p, input_size, img_id_tx, 0, 0)
    if ret:
        return ret

    _, _, result_size = kdp_wrapper.isi_get_result(
        device_index, img_id_tx, 0, 0, inf_res, app_id)

    display_image(inf_res, result_size, frames)

    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()

    return

def image_inference(device_index, app_id, input_size, image_path,
                  img_id_tx, frames):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        input_size: Size of input image.
        ret_size: Return size.
        image_path: image path.
        img_id_tx: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.

    """
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()

    data_p = capture_image(image_path, frames)

    ret, _, img_left = kdp_wrapper.isi_inference(
        device_index, data_p, input_size, img_id_tx, 0, 0)
    if ret:
        return ret

    _, _, result_size = kdp_wrapper.isi_get_result(
        device_index, img_id_tx, 0, 0, inf_res, app_id)

    display_image(inf_res, result_size, frames)

    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()

    return

def tello_inference(device_index, app_id, input_size, frame_read,
                  img_id_tx, frames):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        input_size: Size of input image.
        ret_size: Return size.
        frame_read: Tello capture frame.
        img_id_tx: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.

    """
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()

    data_p = capture_image(frame_read.frame, frames)

    ret, _, img_left = kdp_wrapper.isi_inference(
        device_index, data_p, input_size, img_id_tx, 0, 0)
    if ret:
        return ret

    _, _, result_size = kdp_wrapper.isi_get_result(
        device_index, img_id_tx, 0, 0, inf_res, app_id)

    display_image(inf_res, result_size, frames)

    return

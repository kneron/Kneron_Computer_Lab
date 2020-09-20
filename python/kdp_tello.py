import pygame
from pygame.locals import *
import ctypes
import math
import time
from common import constants
from djitellopy import Tello
from python_wrapper import kdp_wrapper
from python_wrapper import kdp_examples
from kdp_host_api import (kdp_add_dev, kdp_init_log, kdp_lib_init, kdp_lib_start, kdp_lib_de_init)

# Define KL520 parameters
KDP_UART_DEV    = 0
KDP_USB_DEV     = 1
ISI_YOLO_ID     = constants.APP_TINY_YOLO3
IMG_SRC_WIDTH   = 640
IMG_SRC_HEIGHT  = 480
image_size      = IMG_SRC_WIDTH * IMG_SRC_HEIGHT * 2
user_id         = 0
img_id_tx       = 0
frames          = []

# Define Tello parameters
telloSpeed = 50
telloAngle = 30
telloFlag  = 1

# Define PS3 controller parameters
buttonMap   = {'UP':4, 'RIGHT':5, 'DOWN':6, 'LEFT':7, 'L2':8, 'R2':9, 'L1':10,
               'R1':11, 'TRIANGLE':12, 'CIRCLE':13, 'CROSS':14, 'SQUARE':15}

# Initialize KL520 accelerator
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

# Start ISI mode
if (kdp_wrapper.start_isi(dev_idx, ISI_YOLO_ID, IMG_SRC_WIDTH, IMG_SRC_HEIGHT)):
    exit()
        
# Initialize Tello drone
tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

# Initialize PS3 controllers
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Tello flight control
while telloFlag:

    kdp_examples.tello_inference(dev_idx, ISI_YOLO_ID, image_size, frame_read, img_id_tx, frames)

    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
        elif event.type == pygame.JOYBUTTONDOWN:
            if joystick.get_button(buttonMap['R1']):
                tello.takeoff()
            elif joystick.get_button(buttonMap['R2']):
                telloFlag = 0
                tello.land()
                break
            elif joystick.get_button(buttonMap['L1']):
                tello.move_up(telloSpeed)
            elif joystick.get_button(buttonMap['L2']):
                tello.move_down(telloSpeed)
            elif joystick.get_button(buttonMap['TRIANGLE']):
                tello.move_forward(telloSpeed)
            elif joystick.get_button(buttonMap['CROSS']):
                tello.move_back(telloSpeed)
            elif joystick.get_button(buttonMap['CIRCLE']):
                tello.move_right(telloSpeed)
            elif joystick.get_button(buttonMap['SQUARE']):
                tello.move_left(telloSpeed)
            elif joystick.get_button(buttonMap['UP']):
                tello.move_up(telloSpeed)
            elif joystick.get_button(buttonMap['DOWN']):
                tello.move_down(telloSpeed)
            elif joystick.get_button(buttonMap['LEFT']):
                tello.rotate_counter_clockwise(telloAngle)
            elif joystick.get_button(buttonMap['RIGHT']):
                tello.rotate_clockwise(telloAngle)

            img_id_tx += 1

# Exit Tello
tello.streamoff()
tello.end()

# Exit KL520
print("Exit kdp host lib ....\n")
kdp_lib_de_init()

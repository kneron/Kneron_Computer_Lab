
import cv2
import ctypes
import math
from time import sleep
import numpy as np
from common import constants
import kdp_host_api as api

SLEEP_TIME = 0.001
IMG_SOURCE_W = 640
IMG_SOURCE_H = 480
ISI_IMG_SIZE = IMG_SOURCE_W * IMG_SOURCE_H * 2

DME_IMG_SIZE = IMG_SOURCE_W * IMG_SOURCE_H * 2
DME_MODEL_SIZE = 20 * 1024 * 1024
DME_FWINFO_SIZE = 512


def setup_capture(cam_id, width, height):
    """Sets up the video capture device.

    Returns the video capture instance on success and None on failure.

    Arguments:
        width: Width of frames to capture.
        height: Height of frames to capture.
    """
    capture = cv2.VideoCapture(cam_id)
    if not capture.isOpened():
        print("Could not open video device!")
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture

def capture_frame(image):
    if isinstance(image, str):
        print(image)
        frame = cv2.imread(image)

    if isinstance(image, np.ndarray):
        frame = image

    frame = cv2.resize(frame, (IMG_SOURCE_W, IMG_SOURCE_H), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(DME_IMG_SIZE)
    buf_len = DME_IMG_SIZE
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p, buf_len

def start_isi(device_index, app_id, width, height):
    """Starts the ISI mode.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        size: Return size.
        width: Width of the input image.
        height: Height of the input image.
        image_format: Format of input image.
    """
    print("starting ISI mode...\n")
    if (app_id == constants.APP_OD):
        image_format = 0x80000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO  #RGB565, no parallel mode
    else:
        image_format = 0x80000060                                               #RGB565, no parallel mode
    size = 2048

    ret, _, image_buf_size = api.kdp_start_isi_mode(
        device_index, app_id, size, width, height, image_format, 0, 0)
    if ret:
        print("could not set to ISI mode: {} ..\n".format(ret))
        return -1
    if image_buf_size < 3:
        print("ISI mode window {} too small...\n".format(image_buf_size))
        return -1

    print("ISI mode succeeded (window = {})...\n".format(image_buf_size))
    sleep(SLEEP_TIME)
    return 0

def isi_capture_frame(cap, frames):
    """Frame read and convert to RGB565.

    Arguments:
        cap: Active cv2 video capture instance.
        frames: List of frames for the video capture to add to.
    """
    _cv_ret, frame = cap.read()
    if frame is None:
        print("fail to read from cam!")
    frame = cv2.flip(frame, 1)
    frames.append(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame_data = frame.reshape(ISI_IMG_SIZE)
    c_char_p = ctypes.POINTER(ctypes.c_char)
    frame_data = frame_data.astype(np.uint8)
    data_p = frame_data.ctypes.data_as(c_char_p)

    return data_p

def isi_inference(dev_idx, img_buf, buf_len, img_id, rsp_code, window_left):
    """Performs ISI inference.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        img_buf: Image buffer.
        buf_len: File size.
        img_id: Sequence ID of the image.
        rsp_code:
        window_left: Number of image buffers still available for input.
    """
    ret, rsp_code, window_left = api.kdp_isi_inference(
        dev_idx, img_buf, buf_len, img_id, rsp_code, window_left)
    if ret:
        print("ISI inference failed: {}\n".format(ret))
        return -1
    if rsp_code:
        print("ISI inference error_code: [{}] [{}]\n".format(rsp_code, window_left))
        return -1

    return ret, rsp_code, window_left

def isi_get_result(dev_idx, img_id, rsp_code, r_size, r_data, app_id):
    """Gets inference results.

    Arguments:
        dev_idx: Connected device ID. A host can connect several devices.
        img_id: Sequence ID to get inference results of an image with that ID.
        rsp_code:
        r_size: Inference data size.
        r_data: Inference result data.
        app_id: ID of application to be run.
    """
    ret, rsp_code, r_size = api.kdp_isi_retrieve_res(dev_idx, img_id, rsp_code, r_size, r_data)
    if ret:
        print("ISI get [{}] result failed: {}\n".format(img_id, ret))
        return -1, rsp_code, r_size

    if rsp_code:
        print("ISI get [{}] result error_code: [{}] [{}]\n".format(img_id, rsp_code, r_size))
        return -1, rsp_code, r_size

    if r_size >= 4:
        if app_id == constants.APP_AGE_GENDER: # age_gender
            gender = ["Female", "Male"]
            result = ctypes.cast(
                ctypes.byref(r_data), ctypes.POINTER(constants.FDAgeGenderS)).contents
            box_count = result.count
            print("Img [{}]: {} people\n".format(img_id, box_count))
            box = ctypes.cast(
                ctypes.byref(result.boxes),
                ctypes.POINTER(constants.FDAgeGenderRes * box_count)).contents

            for idx in range(box_count):
                print("[{}]: {}, {}\n".format(idx, gender[box[idx].ag_res.ismale], box[idx].ag_res.age))
        else: # od, yolo
            od_header_res = ctypes.cast(
                ctypes.byref(r_data), ctypes.POINTER(constants.ObjectDetectionRes)).contents
            box_count = od_header_res.box_count
            print("image {} -> {} object(s)\n".format(img_id, box_count))

        return 0, rsp_code, r_size
    print("Img [{}]: result_size {} too small\n".format(img_id, r_size))
    return -1, rsp_code, r_size

def sync_inference(device_index, app_id, input_size, capture,
                  img_id_tx, frames, post_handler):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        input_size: Size of input image.
        ret_size: Return size.
        capture: Active cv2 video capture instance.
        img_id_tx: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.
        post_handler: Function to process the results of the inference.
    """
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()

    data_p = isi_capture_frame(capture, frames)

    ret, _, img_left = isi_inference(
        device_index, data_p, input_size, img_id_tx, 0, 0)
    if ret:
        return ret

    _, _, result_size = isi_get_result(
        device_index, img_id_tx, 0, 0, inf_res, app_id)

    post_handler(inf_res, result_size, frames)

    return

def fill_buffer(device_index, capture, size, frames):
    """Fill up the image buffer using the capture device.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        capture: Active cv2 video capture instance.
        size: Size of the input images.
        frames: List of frames captured by the video capture instance.
    """
    print("starting ISI inference ...\n")
    img_id_tx = 1234
    img_left = 12
    buffer_depth = 0
    while 1:
        data_p = isi_capture_frame(capture, frames)
        ret, error_code, img_left = isi_inference(
            device_index, data_p, size, img_id_tx, 0, img_left)
        if ret:
            print("Companion inference failed")
            return -1, img_id_tx, img_left, buffer_depth
        if not error_code:
            img_id_tx += 1
            buffer_depth += 1
            if not img_left:
                break
    return 0, img_id_tx, img_left, buffer_depth

def pipeline_inference(device_index, app_id, loops, input_size, capture,
                  img_id_tx, img_left, buffer_depth, frames, post_handler):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        loops: Number of images to get results.
        input_size: Size of input image.
        ret_size: Return size.
        capture: Active cv2 video capture instance.
        img_id_tx: Should be returned from fill_buffer.
        img_left: Should be returned from fill_buffer.
        buffer_depth: Should be returned from fill_buffer.
        frames: List of frames captured by the video capture instance.
        post_handler: Function to process the results of the inference.
    """
    img_id_rx = 1234
    ret_size = 2048
    inf_res = (ctypes.c_char * ret_size)()
    while loops:
        _, _, result_size = isi_get_result(
            device_index, img_id_rx, 0, 0, inf_res, app_id)
        post_handler(inf_res, result_size, frames)

        img_id_rx += 1
        data_p = isi_capture_frame(capture, frames)

        ret, _, img_left = isi_inference(
            device_index, data_p, input_size, img_id_tx, 0, img_left)
        if ret:
            return ret
        img_id_tx += 1
        loops -= 1

    # Get last 2 results
    while buffer_depth:
        ret, _, result_size = isi_get_result(
            device_index, img_id_rx, 0, 0, inf_res, app_id)
        post_handler(inf_res, result_size, frames)
        img_id_rx += 1
        buffer_depth -= 1
    return 0

def start_isi_parallel(device_index, app_id, width, height):
    """Starts the ISI mode.

    Returns 0 on success and -1 on failure.

    Arguments:
        device_index: Connected device ID. A host can connect several devices.
        app_id: ID of application to be run.
        size: Return size.
        width: Width of the input image.
        height: Height of the input image.
        image_format: Format of input image.
    """
    print("starting ISI mode...\n")
    if (app_id == constants.APP_OD):
        image_format = 0x88000060 | constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO  #RGB565, parallel mode
    else:
        image_format = 0x88000060                                               #RGB565, parallel mode
    size = 2048

    ret, _, image_buf_size = api.kdp_start_isi_mode(
        device_index, app_id, size, width, height, image_format, 0, 0)
    if ret:
        print("could not set to ISI mode: {} ..\n".format(ret))
        return -1
    if image_buf_size < 3:
        print("ISI mode window {} too small...\n".format(image_buf_size))
        return -1

    print("ISI mode succeeded (window = {})...\n".format(image_buf_size))
    sleep(SLEEP_TIME)
    return 0

def kdp_dme_load_model(dev_idx, _model_path):
    """Load dme model."""
    model_id = 0
    data = (ctypes.c_char * DME_FWINFO_SIZE)()
    p_buf = (ctypes.c_char * DME_MODEL_SIZE)()
    ret_size = 0

    # read firmware setup data
    print("loading models to Kneron Device: ")
    n_len = api.read_file_to_buf(data, _model_path + "/fw_info.bin", DME_FWINFO_SIZE)
    if n_len <= 0:
        print("reading fw setup file failed: {}...\n".format(n_len))
        return -1

    dat_size = n_len

    n_len = api.read_file_to_buf(p_buf, _model_path + "/all_models.bin", DME_MODEL_SIZE)
    if n_len <= 0:
        print("reading model file failed: {}...\n".format(n_len))
        return -1

    buf_len = n_len
    model_size = n_len

    print("starting DME mode ...\n")
    ret, ret_size = api.kdp_start_dme(
        dev_idx, model_size, data, dat_size, ret_size, p_buf, buf_len)
    if ret:
        print("could not set to DME mode:{}..\n".format(ret_size))
        return -1

    print("DME mode succeeded...\n")
    print("Model loading successful")
    sleep(SLEEP_TIME)

   # dme configuration
    model_id = 1000  # model id when compiling in toolchain
    output_num = 1     # number of output node for the model
    image_col = 640
    image_row = 480
    image_ch = 3
    image_format = (constants.IMAGE_FORMAT_SUB128 |
                    constants.NPU_FORMAT_RGB565 |
                    constants.IMAGE_FORMAT_RAW_OUTPUT |
                    constants.IMAGE_FORMAT_CHANGE_ASPECT_RATIO)

    dme_cfg = constants.KDPDMEConfig(model_id, output_num, image_col,
                                     image_row, image_ch, image_format)

    dat_size = ctypes.sizeof(dme_cfg)
    print("starting DME configure ...\n")
    ret, model_id = api.kdp_dme_configure(
        dev_idx, ctypes.cast(ctypes.byref(dme_cfg), ctypes.c_char_p), dat_size, model_id)
    if ret:
        print("could not set to DME configure mode..\n")
        return -1

    print("DME configure model [{}] succeeded...\n".format(model_id))
    sleep(SLEEP_TIME)
    return 0

def kdp_inference(dev_idx, img_path):
    """Performs dme inference."""
    img_buf, buf_len = capture_frame(img_path)
    inf_size = 0
    inf_res = (ctypes.c_char * 256000)()
    res_flag = False
    mode = 1
    model_id = 0
    ssid = 0
    status = 0
    _ret, ssid, res_flag = api.kdp_dme_inference(
        dev_idx, img_buf, buf_len, ssid, res_flag, inf_res, mode, model_id)
    # get status for session 1
    while 1:
        status = 0  # Must re-initialize status to 0
        _ret, ssid, status, inf_size = api.kdp_dme_get_status(
            dev_idx, ssid, status, inf_size, inf_res)
        # print(status, inf_size)
        if status == 1:
            npraw_data = get_detection_res(dev_idx, inf_size)
            break
    return npraw_data

def get_detection_res(dev_idx, inf_size):
    """Gets detection results."""
    inf_res = (ctypes.c_char * inf_size)()
    # Get the data for all output nodes: TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) +
    # (H/C/W/RADIX/SCALE) + ... + FP_DATA + FP_DATA + ...
    api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)

    # Prepare for postprocessing
    listdata = [ord(byte) for byte in inf_res]
    npdata = np.asarray(listdata)

    fp_header_res = ctypes.cast(
        ctypes.byref(inf_res), ctypes.POINTER(constants.RawFixpointData)).contents
    output_num = fp_header_res.output_num

    outnode_params_res = ctypes.cast(
        ctypes.byref(fp_header_res.out_node_params),
        ctypes.POINTER(constants.OutputNodeParams * output_num)).contents

    height = 0
    channel = 0
    width = 0
    radix = 0
    scale = 0.0
    npraw_data_array = []
    data_offset = 0
    for param in outnode_params_res:
        height = param.height
        channel = param.channel
        width = param.width
        radix = param.radix
        scale = param.scale

        # print(output_num, height, channel, width, pad_up_16(width), radix, scale)

        # offset in bytes for TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) + (H/C/W/RADIX/SCALE)
        offset = ctypes.sizeof(ctypes.c_int) + output_num * ctypes.sizeof(constants.OutputNodeParams)
        # print("offset ", offset, ctypes.sizeof(c_int), ctypes.sizeof(OutputNodeParams))

        # get the fixed-point data
        npdata = npdata.astype("int8")
        raw_data = []

        raw_data = npdata[offset + data_offset:offset + data_offset + height*channel*pad_up_16(width)]
        data_offset += height*channel*pad_up_16(width)
        # print(raw_data.shape, offset, offset + height*channel*pad_up_16(width), height*channel*pad_up_16(width))
        raw_data = raw_data.reshape(height, channel, pad_up_16(width))
        raw_data = raw_data[:,:,:width]

        # save the fp data into numpy array and convert to float
        npraw_data = np.array(raw_data)
        npraw_data = npraw_data.transpose(0, 2, 1) / (2 ** radix) / scale
        npraw_data_array.append(npraw_data)

    return npraw_data_array

def kdp_exit_dme(dev_idx):
    api.kdp_end_dme(dev_idx)

def pad_up_16(value):
    """Aligns value argument to 16"""
    return math.ceil(value / 16) * 16

def softmax(logits):
    """
    softmax for logits like [[[x1,x2], [y1,y2], [z1,z2], ...]]
    minimum and maximum here work as preventing overflow
    """
    clas = np.exp(np.minimum(logits, 22.))
    clas = clas / np.maximum(np.sum(clas, axis=-1, keepdims=True), 1e-10)
    return clas
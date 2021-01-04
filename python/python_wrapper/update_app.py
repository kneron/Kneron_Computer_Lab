"""
This is the example for update app.
"""
import ctypes
import kdp_host_api as api

HOST_LIB_DIR = ""
TEST_OTA_DIR = "".join([HOST_LIB_DIR, "../app_binaries/ota"])
FW_SCPU_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/fw_scpu.bin"])
FW_NCPU_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/fw_ncpu.bin"])
MODEL_FILE = "".join([TEST_OTA_DIR, "/ready_to_load/model_ota.bin"])

FW_FILE_SIZE = 128 * 1024
MD_FILE_SIZE = 20 * 1024 * 1024

def user_test_app(dev_idx, user_id):
    """User test update firmware"""
    # udt firmware
    module_id = user_id
    img_buf = (ctypes.c_char * FW_FILE_SIZE)()

    if module_id not in (0, 1, 2):
        print("invalid module id: {}...\n".format(user_id))
        return -1

    print("starting update fw ...\n")

    # update scpu
    module_id = 1
    # print(FW_SCPU_FILE)
    buf_len_ret = api.read_file_to_buf(img_buf, FW_SCPU_FILE, FW_FILE_SIZE)

    if buf_len_ret <= 0:
        print("reading scpu file to buf failed: {}...\n".format(buf_len_ret))
    else:
        buf_len = buf_len_ret
        #print("buf len is ", buf_len)
        ret, module_id = api.kdp_update_fw(dev_idx, module_id, img_buf, buf_len)
        if ret:
            print("could not update fw..\n", ret)
        else:
            print("update SCPU firmware succeeded...\n")

    # update ncpu
    buf_len_ret = api.read_file_to_buf(img_buf, FW_NCPU_FILE, FW_FILE_SIZE)
    module_id = 2
    if buf_len_ret <= 0:
        print("reading ncpu file to buf failed: {}...\n".format(buf_len_ret))
    else:
        buf_len = buf_len_ret
        ret, module_id = api.kdp_update_fw(dev_idx, module_id, img_buf, buf_len)
        if ret:
            print("could not update fw..\n", ret)
        else:
            print("update NCPU firmware succeeded...\n")

    # update model
    model_id = 1
    p_buf = (ctypes.c_char * MD_FILE_SIZE)()  # new char[MD_FILE_SIZE];

    print("starting update model: {}...\n".format(model_id))

    buf_len_ret = api.read_file_to_buf(p_buf, MODEL_FILE, MD_FILE_SIZE)
    if buf_len_ret <= 0:
        print("reading model file to buf failed: {}...\n".format(buf_len_ret))
    else:
        buf_len = buf_len_ret
        model_size = buf_len

        ret, model_id = api.kdp_update_model(dev_idx, model_id, model_size, p_buf, buf_len)
        if ret:
            print("could not update model..\n")
        else:
            print("update model succeeded...\n")
    return 0

def user_fw_id(dev_idx):
    """User test get version ID"""
    #print("starting report sys status ...\n")
    ret, sfirmware_id, sbuild_id, _sys_status, _app_status, nfirmware_id, nbuild_id = (
        api.kdp_report_sys_status(dev_idx, 0, 0, 0, 0, 0, 0))
    if ret:
        print("could not report sys status..\n")
        return -1

    print("report sys status succeeded...\n")
    print("\nSCPU firmware_id {}.{}.{}.{} build_id {}\n".format(
        sfirmware_id >> 24, (sfirmware_id & 0x00ff0000) >> 16,
        (sfirmware_id & 0x0000ff00) >> 8,
        (sfirmware_id & 0x000000ff), sbuild_id))
    print("NCPU firmware_id {}.{}.{}.{} build_id {}\n\n".format(
        nfirmware_id >> 24, (nfirmware_id & 0x00ff0000) >> 16,
        (nfirmware_id & 0x0000ff00) >> 8,
        (nfirmware_id & 0x000000ff), nbuild_id))
    return 0

def user_test_update_app(dev_idx, user_id):
    """User test update app"""
    # udt application test
    user_test_app(dev_idx, user_id)
    # udt application id test
    user_fw_id(dev_idx)

    return 0

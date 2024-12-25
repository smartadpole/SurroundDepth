import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '/media/hao/CODE/MODEL/DEPTH/DDAD/onnx/ddad/depth_decoder.onnx'
RKNN_MODEL = '/media/hao/CODE/MODEL/DEPTH/DDAD/onnx/ddad/depth_decoder.rknn'


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)


def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')

def print_onnx(model):
    import onnxruntime as ort

    # 加载 ONNX 模型
    session = ort.InferenceSession(model)

    # 查看模型的输入信息
    print("Model Inputs:")
    for input in session.get_inputs():
        print(f"Name: {input.name}")
        print(f"Shape: {input.shape}")
        print(f"Data Type: {input.type}")
        print("-" * 50)
        break

    # 查看模型的输出信息
    print("Model Outputs:")
    for output in session.get_outputs():
        print(f"Name: {output.name}")
        print(f"Shape: {output.shape}")
        print(f"Data Type: {output.type}")
        print("-" * 50)
        break

if __name__ == '__main__':


    print_onnx(ONNX_MODEL)
    # exit()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.82, 58.82, 58.82], target_platform='rk3588')
    print('done')

    # Load model
    print_onnx(ONNX_MODEL)
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print_onnx(ONNX_MODEL)
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('/media/hao/CODE/RESULT/DDAD/RGB/pred_depth_23.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    start_time = time.time()
    outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
    print('len of output {}'.format(len(outputs)))
    [print('output shape is {}'.format(output.shape)) for output in outputs]
    print('done')
    print("Inference time: {:.2f} seconds".format(time.time() - start_time))
    x = outputs[0]

    rknn.release()
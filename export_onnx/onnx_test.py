#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: onnx_test.py
@time: 2025/1/8 15:15
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))
import argparse
import cv2
import numpy as np
from onnxmodel import ONNXModel
from tools.file import MkdirSimple, match_stereo_file, ReadImageList
import os
import time

DEFAULT_COUNT = 10

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("-o", "--output", type=str, required=True, help="output model path")
    parser.add_argument("--left_image", type=str, default="", help="test image left file or directory")
    parser.add_argument('--right_image', type=str, default="", help="test image right file or directory")

    args = parser.parse_args()
    return args


def inference(img, model):
    c, h, w = model.get_input_size()
    w = w // 2

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    img = img / 255
    img = np.subtract(img, mean)
    img = np.divide(img, std)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    start_time = time.time()
    output = model.forward(img)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    dis_array = np.squeeze(output)
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    return dis_array

def visual_image(img, depth, model):
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_PARULA)
    combined_img = np.vstack((img, depth))

    return combined_img

def test_image(image_files: list, model):
    c, h, w = model.get_input_size()
    # w = w // 2

    results = []
    imgs = []

    for image_file in image_files:
        if image_file == "":
            img = np.clip(np.random.rand(h, w, c) * 255, 0, 255).astype("float32")
        else:
            img = cv2.imread(image_file)
            img = cv2.resize(img, (w, h), cv2.INTER_LANCZOS4)

        imgs.append(img)

    img_org = imgs[0]
    img = np.hstack(imgs)
    depth = inference(img, model)
    combined_img = visual_image(img_org, depth, model)

    return combined_img, depth

def save_image(image, depth, output_dir, file_name):
    depth_file = os.path.join(output_dir, 'depth', file_name)
    concat_file = os.path.join(output_dir, 'concat', file_name)
    MkdirSimple(depth_file)
    MkdirSimple(concat_file)
    cv2.imwrite(concat_file, image)
    cv2.imwrite(depth_file, depth)

def test_dir(model_file, image_dirs, output_dir):
    model = ONNXModel(model_file)
    no_image = not image_dirs or any([not image_dir for image_dir in image_dirs])
    if no_image:
        print("Image path is None, and test with random image")
        print("-" * 50)
        img_lists = [[''] * DEFAULT_COUNT] * max(1, len(image_dirs))
    else:
        img_lists = [ReadImageList(image_dir) for image_dir in image_dirs]

    if not no_image:
        root_len = len(image_dirs[0].strip().rstrip('/'))
    print(f"Test image number: {len(img_lists)} group {len(img_lists[0])} items")

    for img_files in zip(*img_lists):
        image, depth = test_image(img_files, model)
        file_name = img_files[0][root_len+1:] if not no_image else f"random_{time.time()}.jpg"
        save_image(image, depth, output_dir, file_name)

def main():
    args = GetArgs()
    output = args.output
    MkdirSimple(output)

    test_dir(args.model, [args.left_image,], output)


if __name__ == '__main__':
    main()

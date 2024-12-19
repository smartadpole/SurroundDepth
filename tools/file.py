#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: utils.py
@time: 2020/12/4 下午12:28
@desc:
'''
import os
import re

__all__ = ["FILE_SUFFIX", "Walk", "MkdirSimple", "WriteTxt", "WalkImage"]

FILE_SUFFIX = ['jpg', 'png', 'jpeg', 'bmp', 'tiff']


def Walk(path, suffix:tuple):
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if f.endswith(suffix)]

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        file_list.sort()

    return file_list

def WalkImage(path):
    return Walk(path, tuple(FILE_SUFFIX))

def MkdirSimple(path):
    path_current = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if path_current not in ["", "./", ".\\"]:
        os.makedirs(path_current, exist_ok=True)

def WriteTxt(txt, path, encoding):
    MkdirSimple(path)
    with open(path, encoding) as out:
        out.write(txt)
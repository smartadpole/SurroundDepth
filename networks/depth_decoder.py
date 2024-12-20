# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .transformer import *


class DepthDecoder(nn.Module):
    def __init__(self, skip, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, is_train=True):
        super(DepthDecoder, self).__init__()

        self.skip = skip
        self.num_output_channels = num_output_channels
        self.upsample_mode = 'nearest'
        self.scales = list(scales)
        self.use_skips = use_skips
        self.is_train = is_train

        self.iter_num = [8, 8, 8, 8, 8]
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.upconvs0 = nn.ModuleList([ConvBlock(0, 0) for _ in range(5)])
        self.upconvs1 = nn.ModuleList([ConvBlock(0, 0) for _ in range(5)])
        self.dispconvs = nn.ModuleList([ConvBlock(0, 0) for _ in range(5)])
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            self.upconvs0[i] = self.convs[f"upconv_{i}_0"]

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
            self.upconvs1[i] = self.convs[f"upconv_{i}_1"]

        for s in self.scales:
            self.convs[f"dispconv_{s}"] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.dispconvs[s] = self.convs[f"dispconv_{s}"]

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.cross = nn.ModuleList()

        for i in range(len(self.num_ch_enc)):
            self.cross.append(
                CVT(input_channel=self.num_ch_enc[i], downsample_ratio=2 ** (len(self.num_ch_enc) - 1 - i),
                    iter_num=self.iter_num[i]))

        # Initialize the outputs attribute
        if is_train:
            self.outputs = {}
        else:
            self.outputs = torch.Tensor()

    def forward(self, input_features):
        for i, cross in enumerate(self.cross):
            B, C, H, W = input_features[i].shape
            if self.skip:
                input_features[i] = input_features[i] + cross(input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C,
                                                                                                                 H, W)
            else:
                input_features[i] = cross(input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C, H, W)

        # decoder
        x = input_features[-1]
        for i, (dispconv, upconv0, upconv1) in enumerate(zip(self.dispconvs, self.upconvs0, self.upconvs1)):
            x = upconv0(x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = upconv1(x)

            if self.is_train:
                # if i in self.scales:
                #     self.outputs[("disp", i)] = self.sigmoid(dispconv(x))
                pass
            else:
                self.outputs = self.sigmoid(dispconv(x))
        return self.outputs


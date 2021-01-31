#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/01/21 4:28 PM
@description:  
@version: 1.0
"""

import pandas as pd
import json
import os
import argparse
import data_processing as dp

data_path = '/Users/qunyang/Dropbox (Uni of Auckland)/Concrete-shared/' \
            'Qun/ILEE project/Processed data/V2.0'
with open('./data/info/folders.json') as f:
    folders = json.load(f)
with open('./data/info/sensors.json') as f:
    sensors = json.load(f)
save_path = './data/segmented data'
# white_noises = ['W-1', 'W-2', 'W-5', 'W-7',
#                 'W-9', 'W-11', 'W-13',
#                 'W-15', 'W-21', 'W-23']
white_noises = ['W-1']
levels = ['L1', 'L2']
regions = ['Wall', 'Floor']


def seg_signal(args):
    for white_noise in white_noises:
        data = pd.DataFrame()
        signal = dp.Segmentation(data_path, folders, white_noise, args.dim_input)
        for level in levels:
            for region in regions:
                sensor_names = sensors[level][region]
                for sensor_name in sensor_names:
                    data_split = signal(sensor_name)
                    data = pd.concat([data, data_split], axis=1)
        print('Signal segmentation for {} completed.'.format(white_noise))
        if not os.path.exists('./data/segmented data/{}'.format(args.dim_input)):
            os.mkdir('./data/segmented data/{}'.format(args.dim_input))
        data.to_csv('./data/segmented data/{}/{}_segmented.csv'.
                    format(args.dim_input, white_noise), index=None)


def denoise():
    pass


def fft(args):
    for white_noise in white_noises:
        signal = dp.FFT(save_path, white_noise, args.dim_input)
        data = signal()
        print('Signal FFT for {} completed.'.format(white_noise))
        if not os.path.exists('./data/FFT/{}'.format(args.dim_input)):
            os.mkdir('./data/FFT/{}'.format(args.dim_input))
        data.to_csv('./data/FFT/{}/{}_FFT.csv'.
                    format(args.dim_input, white_noise), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_input', default=400, type=int)
    args = parser.parse_args()
    # seg_signal(args)
    fft(args)

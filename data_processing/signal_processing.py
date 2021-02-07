#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/01/21 10:42 AM
@description:  
@version: 1.0
"""


import numpy as np
import pandas as pd
from scipy.fftpack import fft
import json
import sys

if sys.platform == 'win32':
    import matlab
    import matlab.engine


class Segmentation:

    def __init__(self, base_path, folders, white_noise, len_seg):
        self.white_noise = white_noise
        self.data = pd.read_csv('{}/{}/{}-processed.csv'.format(base_path,
                                                                folders[white_noise],
                                                                white_noise),
                                skiprows=[0, 1, 2, 4])
        self.len_seg = len_seg  # Length of segmentation

    def __call__(self, *args, **kwargs):
        data, num = self.write_signal(args[0])
        return data, num

    def seg_signal(self, sensor_name):
        """
        Segment signal
        :param sensor_name: Sensor name eg. A-L2-Floor-B2-EW
        :return: Segmented signal
        """
        data = np.array(self.data[sensor_name])
        num_seg = len(data) // self.len_seg + 1
        data_split = []
        idx = 0
        for i in range(num_seg):
            if i < num_seg - 1:
                tmp = data[idx: idx + self.len_seg]
                idx += self.len_seg
            else:
                tmp = np.pad(data[idx:],
                             (0, self.len_seg - (len(data) % self.len_seg)),
                             'constant', constant_values=0)
            data_split.append(tmp)
        return data_split, num_seg

    def write_signal(self, sensor_name):
        """

        :param sensor_name:
        :return:
        """
        data = pd.DataFrame()
        signals, num_seg = self.seg_signal(sensor_name)
        for idx, signal in enumerate(signals):
            data.insert(idx, '{}_{}'.format(sensor_name, idx + 1), signal)
        return data, num_seg


class Denoise:
    
    def __init__(self, base_path, white_noise, len_seg):
        self.white_noise = white_noise
        self.data = pd.read_csv('{}/{}/{}_segmented.csv'.format(base_path,
                                                                len_seg,
                                                                white_noise),
                                )

    def __call__(self, *args, **kwargs):
        data = self.wavelet_denoise()
        return data

    @staticmethod
    def convert(x):
        c = []
        for i in range(x.size[1]):
            c.append(x._data[i * x.size[0]: i * x.size[0] + x.size[0]][0])
        return c

    def wavelet_denoise(self):
        data = pd.DataFrame()
        engine = matlab.engine.start_matlab()
        for idx, header in enumerate(self.data.columns):
            noisy_signal = matlab.double(list(self.data[header]))
            denoised_signal = engine.denoise(noisy_signal)
            denoised_signal = self.convert(denoised_signal)
            data.insert(idx, header, denoised_signal)
        return data


class FFT:

    def __init__(self, base_path, white_noise, len_seg):
        self.white_noise = white_noise
        self.data = pd.read_csv('{}/{}/{}_denoised.csv'.format(base_path,
                                                               len_seg,
                                                               white_noise),
                                )

    def __call__(self, *args, **kwargs):
        data = self.write_fft()
        return data

    @staticmethod
    def fft(data, sampling_rate=256):
        num_freqs = int(2 ** np.ceil(np.log2(len(data))))
        freq_domain = (sampling_rate / 2) * np.linspace(0, 1, int(num_freqs / 2))  # Frequency domain
        amplitude = fft(data, num_freqs)  # FFT
        amplitude = 2 * np.abs(amplitude[0: int(len(freq_domain) / 2)])
        amplitude[0] = 0
        freq_domain_half = freq_domain[0: int(len(freq_domain) / 2)]
        norm_amplitude = amplitude / np.max(amplitude)
        return freq_domain_half, norm_amplitude

    def write_fft(self):
        data = pd.DataFrame()
        for idx, header in enumerate(self.data.columns):
            freq_domain, norm_amplitude = self.fft(self.data[header])
            data.insert(idx, header, norm_amplitude)
        data.insert(0, 'Frequency', freq_domain)
        return data


if __name__ == '__main__':
    data_path = '/Users/qunyang/Dropbox (Uni of Auckland)/Concrete-shared/' \
                'Qun/ILEE project/Processed data/V2.0'
    with open('../data/info/folders.json') as f:
        folders = json.load(f)
    seg = Segmentation(data_path, folders, 'W-1', 400)
    a = seg.write_signal('A-L2-Floor-B2-EW')
    path = '../data/segmented data'
    f = FFT(path, 'W-1', 400)
    b = f.write_fft()

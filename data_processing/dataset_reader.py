#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 10:07 AM
@description:  
@version: 1.0
"""


import numpy as np


class DatasetReader:

    def __init__(self, white_noise, data_path, len_seg):
        self.white_noise = white_noise
        self.data = np.load('{}/FFT/{}/{}_FFT.npy'.format(data_path, len_seg, white_noise))
        self.dataset, self.dataset_ = self.generate_dataset()

    def __call__(self, *args, **kwargs):
        print('Preparing {} dataset...'.format(self.white_noise))
        dataset = self.generate_dataset()
        return dataset

    def generate_dataset(self):
        """
        [12, 3, num_seg, num_feature] => [12 * num_seg, num_feature * 3]
        num_feature = 128
        :return:
        """
        num_sensor = self.data.shape[0]
        num_channel = self.data.shape[1]
        num_seg = self.data.shape[2]
        num_feature = self.data.shape[3]
        # [12, 3, num_seg, num_feature] => [12, num_seg, num_feature * 3]
        dataset_ = np.zeros((num_sensor, num_seg, num_feature * num_channel))
        for i in range(num_sensor):
            tmp = self.data[i][0]
            for j in range(1, num_channel):
                tmp = np.hstack((tmp, self.data[0][j]))
            dataset_[i] = tmp
        # [12, num_seg, num_feature * 3] => [12 * num_seg, num_feature * 3]
        dataset = dataset_.reshape((num_sensor * num_seg, -1))
        return dataset, dataset_


if __name__ == '__main__':
    data_path = '../data/data_processed'
    reader = DatasetReader('W-1', data_path, 400)
    dataset = reader()

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


import torch
import numpy as np


class DatasetReader:

    def __init__(self, white_noise, data_path, data_source, len_seg):
        self.white_noise = white_noise
        self.data = np.load('{0}/{1}/{2}/{3}_{1}.npy'.format(data_path, data_source, len_seg, white_noise))
        self.num_sensor = self.data.shape[0]
        self.num_channel = self.data.shape[1]
        self.num_seg = self.data.shape[2]
        self.num_feature = self.data.shape[3]

    def __call__(self, *args, **kwargs):
        print('Preparing {} dataset...'.format(self.white_noise))
        if args[0] == 'MLP':
            trainset, testset = self.gen_trainset_mlp(), self.gen_testset_mlp()
        else:
            trainset, testset = self.gen_trainset_conv2d(), self.gen_testset_conv2d()
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        return torch.from_numpy(trainset), torch.from_numpy(testset)

    def gen_trainset_mlp(self):
        """
        [12, 3, num_seg, num_feature] => [12 * num_seg, num_feature * 3]
        num_feature = 128
        :return:
        """
        trainset = self.gen_testset_mlp()
        # [12, num_seg, num_feature * 3] => [12 * num_seg, num_feature * 3]
        trainset = trainset.reshape((self.num_sensor * self.num_seg, -1))
        return trainset

    def gen_testset_mlp(self):
        # [12, 3, num_seg, num_feature] => [12, num_seg, num_feature * 3]
        testset = np.zeros((self.num_sensor, self.num_seg, self.num_feature * self.num_channel))
        for i in range(self.num_sensor):
            tmp = self.data[i][0]
            for j in range(1, self.num_channel):
                tmp = np.hstack((tmp, self.data[i][j]))
            testset[i] = tmp
        return testset

    def gen_trainset_conv2d(self):
        # [12, 3, num_seg, num_feature] => [12 * num_seg, 3, num_feature]
        trainset = np.zeros((self.num_sensor * self.num_seg, self.num_channel, self.num_feature))
        m = 0
        for i in range(self.num_sensor):
            for j in range(self.num_seg):
                for k in range(self.num_channel):
                    trainset[m][k][:] = self.data[i][k][j][:]
                m += 1
        return trainset

    def gen_testset_conv2d(self):
        # [12, 3, num_seg, num_feature] => [12, num_seg, 3, num_feature]
        return self.data.transpose((0, 2, 1, 3))


if __name__ == '__main__':
    data_path = '../data/data_processed'
    reader = DatasetReader('W-1', data_path, 400)
    trainset, testset = reader('MLP')

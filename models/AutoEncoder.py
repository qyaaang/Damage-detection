#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/02/21 11:59 AM
@description:  
@version: 1.0
"""


from torch import nn
from layers.AE import AutoEncoder, AutoEncoderConv


net_classes = {'MLP': AutoEncoder,
               'Conv2D': AutoEncoderConv
               }


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        net = net_classes[args.net_name]
        self.AE = net(args)

    def forward(self, x):
        if self.args.model_name == 'VAE':
            output, z, kld = self.AE(x)
            return output, z, kld
        else:
            output, z = self.AE(x)
            return output, z

#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 3:05 PM
@description:  
@version: 1.0
"""


from torch import nn
from layers.MLP import Gen_MLP


net_classes = {'MLP': Gen_MLP}


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        net = net_classes[args.net_name]
        self.net = net(dim_noise=args.dim_noise,
                       dim_hidden=args.dim_hidden,
                       dim_output=args.dim_output
                       )

    def forward(self, z):
        output = self.net(z)
        return output

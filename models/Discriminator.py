#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 3:23 PM
@description:  
@version: 1.0
"""


from torch import nn
from layers.MLP import MLP_D
from layers.Conv2D import Conv2D_D


net_classes = {'MLP': MLP_D,
               'Conv2D': Conv2D_D
               }


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        net = net_classes[args.net_name]
        if args.net_name == 'MLP':
            self.net = net(dim_input=args.dim_input,
                           dim_hidden=args.dim_hidden,
                           )
        else:
            self.net = net()

    def forward(self, x):
        output = self.net(x)
        return output

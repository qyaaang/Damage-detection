#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 11:03 AM
@description:  
@version: 1.0
"""


import torch
from torch.utils.data import DataLoader
from torch import nn, optim, autograd
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
import time
import argparse
from models.Generator import Generator
from models.Discriminator import Discriminator


data_path = './data/data_processed'
save_path = './results'


class BaseExperiment:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        print('> Training arguments:')
        for arg in vars(args):
            print('>>> {}: {}'.format(arg, getattr(args, arg)))
        white_noise = dp.DatasetReader(white_noise=self.args.dataset,
                                       data_path=data_path,
                                       data_source=args.data,
                                       len_seg=self.args.len_seg
                                       )
        dataset, _ = white_noise(args.net_name)
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      )
        self.Generator = Generator(args)  # Generator
        self.Discriminator = Discriminator(args)  # Discriminator

    def select_optimizer(self, model):
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=self.args.learning_rate,
                                   betas=(0.5, 0.9)
                                   )
        elif self.args.optimizer == 'RMS':
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=self.args.learning_rate
                                      )
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=self.args.learning_rate,
                                  momentum=0.9
                                  )
        elif self.args.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=self.args.learning_rate
                                      )
        elif self.args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=self.args.learning_rate
                                       )
        return optimizer

    @staticmethod
    def weights_init(m):
        """
        Custom weights initialization called on netG and netD
        :param m:
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                          self.args.net_name,
                                          self.args.len_seg,
                                          self.args.optimizer,
                                          self.args.learning_rate,
                                          self.args.num_epoch
                                          )

    def gradient_penalty(self, x_real, x_fake, batch_size, beta=0.3):
        x_real = x_real.detach()
        x_fake = x_fake.detach()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(x_real)
        interpolates = alpha * x_real + ((1 - alpha) * x_fake)
        interpolates.requires_grad_()
        dis_interpolates = self.Discriminator(interpolates)
        gradients = autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(dis_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * beta
        return grad_penalty

    def train(self):
        # self.Generator.apply(self.weights_init)
        self.weights_init(self.Generator)
        self.weights_init(self.Discriminator)
        optimizerD = self.select_optimizer(self.Generator)
        optimizerG = self.select_optimizer(self.Generator)
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(64, 100, 1, 1)
        c = nn.MSELoss()
        real_label = 1.
        fake_label = 0.
        G_losses = []
        D_losses = []
        for epoch in range(self.args.num_epoch):
            t0 = time.time()
            for i, sample_batched in enumerate(self.data_loader):
                # 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                self.Discriminator.zero_grad()
                data_real = torch.tensor(sample_batched, dtype=torch.float32)
                data_real = data_real.unsqueeze(2)
                batch_size = sample_batched.size(0)
                label = torch.full((batch_size, ), 1, dtype=torch.float32)
                output = self.Discriminator(data_real)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                # Generate data
                noise = torch.rand(batch_size, 100, 1, 1)
                fake = self.Generator(noise).detach()
                label.fill_(fake_label)
                output = self.Discriminator(fake)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                optimizerD.zero_grad()
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()
                # Train Generator: maximize log(D(G(z)))
                self.Generator.zero_grad()
                label.fill_(real_label)
                output = self.Discriminator(fake)
                errG = criterion(output, label)
                optimizerG.zero_grad()
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                # mse = criterion(fake, data_real)
                f = fake.squeeze(2)
                r = data_real.squeeze(2)
                mse = c(f, r)
            # t1 = time.time()
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.args.num_epoch, i, len(self.data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    print(mse)
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                data_fake = f.numpy()
                data_real = r.numpy()
            fig, ax = plt.subplots()
            ax.plot(data_fake[0][1], label='fake')
            ax.plot(data_real[0][1], ls='--', lw=0.5, label='real')
            ax.legend()
        plt.show()


def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='W-1', type=str)
    parser.add_argument('--data', default='FFT', type=str)
    parser.add_argument('--model_name', default='GAN', type=str)
    parser.add_argument('--net_name', default='Conv2D', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--dim_noise', default=50, type=int)
    parser.add_argument('--dim_input', default=384, type=int)
    parser.add_argument('--dim_hidden', default=1000, type=int)
    parser.add_argument('--dim_output', default=384, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    args = parser.parse_args()
    exp = BaseExperiment(args)
    exp.train()


if __name__ == '__main__':
    main()

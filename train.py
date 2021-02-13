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
from torch import nn, optim
from adabelief_pytorch import AdaBelief
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
import time
import json
import argparse
from models.AutoEncoder import AutoEncoder
import visdom


data_path = './data/data_processed'
info_path = './data/info'
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
                                       data_source=args.data_source,
                                       len_seg=self.args.len_seg
                                       )
        dataset, _ = white_noise(args.net_name)
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False
                                      )
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.AE = AutoEncoder(args)  # AutoEncoder
        self.AE.apply(self.weights_init)
        self.criterion = nn.MSELoss()
        self.vis = visdom.Visdom()

    def select_optimizer(self, model):
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=self.args.learning_rate,
                                   betas=(0.5, 0.9),
                                   )
        elif self.args.optimizer == 'AdaBelief':
            optimizer = AdaBelief(model.parameters(),
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

    def weights_init(self, m):
        initializers = {'xavier_uniform_': nn.init.xavier_uniform_,
                        'xavier_normal_': nn.init.xavier_normal,
                        'orthogonal_': nn.init.orthogonal_,
                        'kaiming_normal_': nn.init.kaiming_normal_
                        }
        initializer = initializers[self.args.initializer]
        if isinstance(m, nn.Linear):
            initializer(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                          self.args.net_name,
                                          self.args.len_seg,
                                          self.args.optimizer,
                                          self.args.learning_rate,
                                          self.args.num_epoch
                                          )

    def train(self):
        optimizer = self.select_optimizer(self.AE)
        best_loss = 100.
        best_epoch = 1
        lh = {}
        losses, mses, klds = [], [], []
        for epoch in range(self.args.num_epoch):
            t0 = time.time()
            if self.args.net_name == 'MLP':
                if self.args.model_name == 'VAE':
                    f = torch.zeros(len(self.data_loader.dataset), int(self.args.dim_feature / 2))
                else:
                    f = torch.zeros(len(self.data_loader.dataset), int(self.args.dim_feature))
            else:
                f = torch.zeros(len(self.data_loader.dataset), self.args.num_hidden_map, 1, 8)
            idx = 0
            for _, sample_batched in enumerate(self.data_loader):
                batch_size = sample_batched.size(0)
                x = torch.tensor(sample_batched, dtype=torch.float32)
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(2)
                if self.args.model_name == 'VAE':
                    x_hat, z, z_kld = self.AE(x)
                    loss = self.criterion(x_hat, x)
                    elbo = - loss - 1.0 * z_kld
                    loss = - elbo
                else:
                    x_hat, z = self.AE(x)
                    mse = self.criterion(x_hat, x)
                    kld = torch.sum(x * torch.log(x + 1e-7 / x_hat)) / x.numel()
                    loss = mse + self.args.err_kld_coef * kld
                    loss = loss
                f[idx: idx + batch_size] = z
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                idx += batch_size
            t1 = time.time()
            if self.args.model_name == 'VAE':
                print('\033[1;31mEpoch: {}\033[0m\t'
                      '\033[1;32mReconstruction loss: {:5f}\033[0m\t'
                      '\033[1;33mKL Divergence: {:5f}\033[0m\t'
                      'Time cost: {:2f}s'
                      .format(epoch + 1, loss.item(), z_kld, t1 - t0))
            else:
                print('\033[1;31mEpoch: {}\033[0m\t'
                      '\033[1;32mReconstruction loss: {:5f}\033[0m\t'
                      '\033[1;33mMSE: {:5f}\033[0m\t'
                      '\033[1;34mKL Divergence: {:2f}\033[0m\t'
                      'Time cost: {:2f}s'
                      .format(epoch + 1, loss.item(), mse.item(), kld.item(), t1 - t0))
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch + 1
                f = f.detach().numpy()
                path = '{}/models/{}.model'.format(save_path, self.file_name())
                torch.save(self.AE.state_dict(), path)
                np.save('{}/features/{}.npy'.format(save_path, self.file_name()), f)
            losses.append(loss.item())
            mses.append(mse.item())
            klds.append(kld.item())
            self.show_loss(loss, epoch)
            self.show_reconstruction(epoch)
        lh['Loss'] = losses
        lh['MSE'] = mses
        lh['KL Divergence'] = klds
        lh['Min loss'] = best_loss
        lh['Best epoch'] = best_epoch
        lh = json.dumps(lh, indent=2)
        with open('{}/learning history/{}.json'.format(save_path, self.file_name()), 'w') as f:
            f.write(lh)

    def show_loss(self, loss, epoch):
        self.vis.line(Y=np.array([loss.item()]), X=np.array([epoch + 1]),
                      win='Train loss',
                      opts=dict(title='Train loss'),
                      update='append'
                      )

    def show_reconstruction(self, epoch, seg_idx=25):
        plt.figure(figsize=(15, 15))
        num_seg = int(self.data_loader.dataset.shape[0] / len(self.spots))
        spots_l1, spots_l2 = np.hsplit(self.spots, 2)
        for i, (spot_l1, spot_l2) in enumerate(zip(spots_l1, spots_l2)):
            # L1 sensors
            plt.subplot(int(len(self.spots) / 2), 2, 2 * i + 1)
            x = torch.tensor(self.data_loader.dataset[i * num_seg + seg_idx], dtype=torch.float32)
            if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
            plt.plot(x.view(-1).detach().numpy(), label='original')
            plt.title('A-{}-{}'.format(spot_l1, seg_idx))
            x_hat, _ = self.AE(x)
            plt.plot(x_hat.view(-1).detach().numpy(), label='reconstruct')
            plt.axvline(x=127, ls='--', c='k')
            plt.axvline(x=255, ls='--', c='k')
            plt.legend(loc='upper center')
            # L2 sensors
            plt.subplot(int(len(self.spots) / 2), 2, 2 * (i + 1))
            x = torch.tensor(self.data_loader.dataset[(i + 5) * num_seg + seg_idx],
                             dtype=torch.float32)
            if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
            plt.plot(x.view(-1).detach().numpy(), label='original')
            plt.title('A-{}-{}'.format(spot_l2, seg_idx))
            x_hat, _ = self.AE(x)
            plt.plot(x_hat.view(-1).detach().numpy(), label='reconstruct')
            plt.axvline(x=127, ls='--', c='k')
            plt.axvline(x=255, ls='--', c='k')
            plt.legend(loc='upper center')
        plt.subplots_adjust(hspace=0.5)
        self.vis.matplot(plt, win='Reconstruction', opts=dict(title='Epoch: {}'.format(epoch + 1)))
        plt.close()

def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='W-1', type=str)
    parser.add_argument('--data_source', default='FFT', type=str)
    parser.add_argument('--model_name', default='AE', type=str)
    parser.add_argument('--net_name', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--initializer', default='xavier_normal_', type=str)
    # MLP setting
    parser.add_argument('--dim_input', default=384, type=int)
    parser.add_argument('--dim_feature', default=20, type=int)
    # Conv2D setting
    parser.add_argument('--num_feature_map', default=128, type=int)
    parser.add_argument('--num_hidden_map', default=256, type=int)
    parser.add_argument('--err_kld_coef', default=0.0, type=float)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    args = parser.parse_args()
    exp = BaseExperiment(args)
    exp.train()


if __name__ == '__main__':
    main()

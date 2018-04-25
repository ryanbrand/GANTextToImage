from __future__ import print_function
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p

from tensorboard import summary
from tensorboard import FileWriter

from model import G_NET, D_NET64, D_NET128, D_NET256, INCEPTION_V3

def save_superimages(images_list, save_dir, imsize):
    batch_size = images_list[0].size(0)
    num_sentences = len(images_list)
    for i in range(batch_size):
        s_tmp = '%s/super/' %\
            (save_dir)
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        #
        savename = '%s_%d.png' % (s_tmp, imsize)
        super_img = []
        for j in range(num_sentences):
            img = images_list[j][i]
            # print(img.size())
            img = img.view(1, 3, imsize, imsize)
            # print(img.size())
            super_img.append(img)
            # break
        super_img = torch.cat(super_img, 0)
        vutils.save_image(super_img, savename, nrow=10, normalize=True)

def save_singleimages(images, save_dir, sentenceID, imsize):
    for i in range(images.size(0)):
        s_tmp = '%s/single_samples' %\
            (save_dir)
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
        # range from [-1, 1] to [0, 255]
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def test(cfg_file, embedding_t7_path):

    cfg_from_file(cfg_file)

    cfg.GPU_ID = 0

    print('Using config:')
    pprint.pprint(cfg)

    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=[0])
    print(netG)
    #state_dict = torch.load('../gan/models/birds_3stages/netG_26000.pth')
    print(cfg.TRAIN.NET_G)
    state_path = '/home/ubuntu/GANTextToImage/gan/' + cfg.TRAIN.NET_G
    state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    # print('Load ', '../gan/models/flowers_1stage/netG_6000.pth')

    # the path to save generated images
    s_tmp = cfg.TRAIN.NET_G
    istart = s_tmp.rfind('_') + 1
    iend = s_tmp.rfind('.')
    iteration = int(s_tmp[istart:iend])
    s_tmp = s_tmp[:s_tmp.rfind('/')]
    save_dir = '%s/iteration%d' % (s_tmp, iteration)

    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(1, nz))
    if cfg.CUDA:
        netG.cuda()
        noise = noise.cuda()

    # switch to evaluate mode
    netG.eval()

    t_embedding = load_lua(embedding_t7_path)
    t_embedding = t_embedding.unsqueeze(0)
    print(t_embedding.size())   

    if cfg.CUDA:
        t_embedding = Variable(t_embedding).cuda()
    else:
        t_embedding = Variable(t_embedding)
    # print(t_embeddings[:, 0, :], t_embeddings.size(1))

    embedding_dim = t_embedding.size(1)
    noise.data.resize_(1, nz)
    noise.data.normal_(0, 1)

    fake_img_list = []
    # for i in range(embedding_dim):
    fake_imgs, _, _ = netG(noise, t_embedding[:, 0, :])
    if cfg.TEST.B_EXAMPLE:
            # fake_img_list.append(fake_imgs[0].data.cpu())
            # fake_img_list.append(fake_imgs[1].data.cpu())
        fake_img_list.append(fake_imgs[2].data.cpu())
    else:
        save_singleimages(fake_imgs[-1], '/home/ubuntu/GANTextToImage/static', 0, 256)
            # self.save_singleimages(fake_imgs[-2], filenames,
            #                        save_dir, split_dir, i, 128)
            # self.save_singleimages(fake_imgs[-3], filenames,
            #                        save_dir, split_dir, i, 64)
        # break
    # if cfg.TEST.B_EXAMPLE:
    #     # self.save_superimages(fake_img_list, filenames,
    #     #                       save_dir, split_dir, 64)
    #     # self.save_superimages(fake_img_list, filenames,
    #     #                       save_dir, split_dir, 128)
    #     save_superimages(fake_img_list, '.', 256)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate images from an embedding.')
    parser.add_argument('embedding_file_path', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='../gan/cfg/eval_birds.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='.')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
 
    args = parser.parse_args()
 
    test(args.cfg_file, args.embedding_file_path)

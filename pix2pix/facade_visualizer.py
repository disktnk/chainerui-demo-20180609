#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

from chainerui import summary

def out_image(updater, enc, dec, n_images, rows, seed):
    def make_image(trainer):
        np.random.seed(seed)
        xp = enc.xp
        
        w_in = 256
        w_out = 256
        in_ch = 12
        out_ch = 3
        
        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("i")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        
        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)

            z = enc(x_in)
            x_out = dec(z)
            
            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]

        summary.image(gen_all, name='gen', row=rows)

        x = np.ones((n_images, 3, w_in, w_in)).astype(np.uint8)*255
        x[:,0,:,:] = 0
        for i in range(12):
            x[:,0,:,:] += np.uint8(15*i*in_all[:,i,:,:])
        summary.image(x, name='in', row=rows, mode='HSV')

        summary.image(gt_all, name='gt', row=rows)
        
    return make_image

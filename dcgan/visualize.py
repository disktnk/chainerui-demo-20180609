#!/usr/bin/env python

import numpy as np

import chainer
import chainer.cuda
from chainer import Variable

from chainerui import summary


def out_generated_image(gen, dis, rows, cols, seed):
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        summary.image(x, row=rows)
        np.random.seed()
    return make_image

# VAE (MNIST) with ChainerUI

original code: [chainerui/examples](https://github.com/chainer/chainerui/tree/master/examples)

```bash
$ chainerui project create -d results -n vae-mnist
```

```bash
$ python train_vae.py -o results/demo -g 0
```

## Usage

from `train_vae.py`:

```py
from chainerui import summary

def main()
    # ...(snip)
    # [chainerui] add extension with visualizer
    #             collect latest image from cache area
    @chainer.training.make_extension()
    def out_generated_image(trainer):
        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        x = chainer.Variable(np.asarray(train[train_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)

        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        x_test = chainer.Variable(np.asarray(test[test_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1_test = model(x)

        # draw images from randomly sampled z
        z = chainer.Variable(
            np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
        x_sampled = model.decode(z)

        epoch = trainer.updater.epoch
        iteration = trainer.updater.iteration
        with summary.reporter(args.out, epoch=epoch, iteration=iteration) as r:
            r.image(x.reshape(len(train_ind), 28, 28), 'train', row=3)
            r.image(x1.reshape(len(train_ind), 28, 28), 'train_reconstructed',
                    row=3)
            r.image(x_test.reshape(len(test_ind), 28, 28), 'test', row=3)
            r.image(x1_test.reshape(len(test_ind), 28, 28),
                    'test_reconstructed', row=3)
            r.image(x_sampled.reshape(9, 28, 28), 'sampled', row=3)
    trainer.extend(out_generated_image, trigger=(1, 'epoch'))
```

on ChainerUI:

![web image](sample_vae.png)

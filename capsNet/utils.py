import os

import numpy as np
import scipy
import tensorflow as tf
from capsNet.config import Config as conf


def load_from_file(path, mode):
    if path.startswith('gs://'):  # For google cloud ml-engine. pickle
        from tensorflow.python.lib.io import file_io
        f = file_io.FileIO(path + '.npy', mode)
        loaded = np.load(f)
    else:
        fd = open(path, mode)
        loaded = np.fromfile(file=fd, dtype=np.uint8)
    return loaded


def load_mnist(path, is_training):
    loaded = load_from_file(os.path.join(conf.dataset, 'train-images-idx3-ubyte'), 'r')
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 'train-labels-idx1-ubyte'), 'r')
    trY = loaded[8:].reshape((60000)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 't10k-images-idx3-ubyte'), 'r')
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    loaded = load_from_file(os.path.join(conf.dataset, 't10k-labels-idx1-ubyte'), 'r')
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)

    # => [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX / 255., teY


def get_batch_data():
    trX, trY = load_mnist(conf.dataset, is_training=True)

    num_batch = int(trX.get_shape()[0] // conf.batch_size)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=conf.batch_size,
                                  capacity=conf.batch_size * 64,
                                  min_after_dequeue=conf.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return X, Y, num_batch


def save_images(imgs, size, path):
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(conf.dataset, is_training=True)
    print(X.get_shape())
    print(X.dtype)
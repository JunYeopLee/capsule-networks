from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from config import Config as conf
from model import CapsNet
from utils import load_mnist, save_images


def main(_):
    # Load Graph
    capsNet = CapsNet(is_training=False)
    print('[+] Graph is constructed')

    # Load test data
    teX, teY = load_mnist(conf.dataset, is_training=False)

    # Start session
    with capsNet.graph.as_default():
        sv = tf.train.Supervisor(logdir=conf.logdir)
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            checkpoint_path = tf.train.latest_checkpoint(conf.logdir)
            sv.saver.restore(sess, checkpoint_path)
            print('[+] Graph is restored from ' + checkpoint_path)

            # Make results directory
            if not os.path.exists('results'):
                os.mkdir('results')

            reconstruction_err = []
            classification_acc = []
            for i in range(10000 // conf.batch_size):
                start = i * conf.batch_size
                end = start + conf.batch_size

                # Reconstruction
                recon_imgs = sess.run(capsNet.decoded, {capsNet.x: teX[start:end]})
                recon_imgs = np.reshape(recon_imgs, (conf.batch_size, -1))
                orgin_imgs = np.reshape(teX[start:end], (conf.batch_size, -1))
                squared = np.square(recon_imgs - orgin_imgs)
                reconstruction_err.append(np.mean(squared))
                if i % 5 == 0:
                    imgs = np.reshape(recon_imgs, (conf.batch_size, 28, 28, 1))
                    size = 6
                    save_images(imgs[0:size * size, :], [size, size], 'results/test_%03d.png' % i)

                # Classification
                cls_result = sess.run(capsNet.preds, {capsNet.x: teX[start:end]})
                cls_answer = teY[start:end]
                cls_acc = np.mean(np.equal(cls_result, cls_answer).astype(np.float32))
                classification_acc.append(cls_acc)

            # Print classification accuracy & reconstruction error
            print('reconstruction_err : ' + str(np.mean(reconstruction_err)))
            print('classification_acc : ' + str(np.mean(classification_acc) * 100))


if __name__ == "__main__":
    tf.app.run()

import tensorflow as tf

from config import Config as conf
from utils import *


class CapsNet:
    def __init__(self, is_training=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
            self.build_model(is_training)

    def build_model(self, is_training=False):
        # Input Layer, Reshape to [batch, height, width, channel]
        self.input = tf.reshape(self.x, [conf.batch_size, 28, 28, 1])
        assert self.input.get_shape() == (conf.batch_size, 28, 28, 1)

        # ReLU Conv1
        with tf.variable_scope('conv1_layer'):
            conv1 = tf.layers.conv2d(
                inputs=self.input,
                filters=256,
                kernel_size=9,
                activation=tf.nn.relu,
            )
            assert conv1.get_shape() == (conf.batch_size, 20, 20, 256)

        # Primary Caps
        with tf.variable_scope('primary_caps'):
            pcaps = self.primary_caps(
                inputs=conv1
            )
            assert pcaps.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Digit Caps
        with tf.variable_scope('digit_caps'):
            dcaps = self.digit_caps(
                inputs=pcaps,
                num_iters=3,
                num_caps=10,
            )
            assert dcaps.get_shape() == (conf.batch_size, 10, 16)

        # Prediction
        with tf.variable_scope('prediction'):
            self.logits = tf.sqrt(tf.reduce_sum(tf.square(dcaps), axis=2) + conf.eps)  # [batch_size, 10]
            self.probs = tf.nn.softmax(self.logits)
            self.preds = tf.argmax(self.probs, axis=1)  # [batch_size]
            assert self.logits.get_shape() == (conf.batch_size, 10)

        # Reconstruction
        with tf.variable_scope('reconstruction'):
            targets = tf.argmax(self.y, axis=1) if is_training else self.preds
            self.decoded = self.reconstruction(
                inputs=dcaps,
                targets=targets,
            )
            assert self.decoded.get_shape() == (conf.batch_size, 28, 28)

        if not is_training: return

        # Margin Loss
        with tf.variable_scope('margin_loss'):
            self.mloss = self.margin_loss(
                logits=self.logits,
                labels=self.y,
            )

        # Reconstruction Loss
        with tf.variable_scope('reconsturction_loss'):
            self.rloss = self.reconstruction_loss(
                origin=self.x,
                decoded=self.decoded,
            )

        # Train
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss = self.mloss + 0.0005 * self.rloss
        self.train_vars = [(v.name, v.shape) for v in tf.trainable_variables()]
        self.train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Summary
        tf.summary.scalar('margin_loss', self.mloss)
        tf.summary.scalar('reconstruction_loss', self.rloss)
        tf.summary.scalar('total_loss', self.loss)
        self.summary = tf.summary.merge_all()

        return

    def primary_caps(self, inputs):
        # Validate inputs
        assert inputs.get_shape() == (conf.batch_size, 20, 20, 256)

        # Convolution
#        convs = []
#        for i in range(32):
#            conv = tf.layers.conv2d(
#                inputs=inputs,
#                filters=8,
#                kernel_size=9,
#                strides=2,
#            )
#            assert conv.get_shape() == (conf.batch_size, 6, 6, 8)
#            flat_shape = (conf.batch_size, 6*6, 8)
#            conv_flatten = tf.reshape(conv, flat_shape)
#            convs.append(conv_flatten)
#        convs = tf.concat(convs, axis=1)
#        assert convs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Convolution (batched)
        convs = tf.layers.conv2d(
            inputs=inputs,
            filters=32*8,
            kernel_size=9,
            strides=2,
            activation=tf.nn.relu,
        )
        convs = tf.reshape(convs, [conf.batch_size, -1, 8])
        assert convs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Squash
        pcaps = self.squash(convs)
        return pcaps

    def digit_caps(self, inputs, num_iters, num_caps):
        # Validate inputs
        assert inputs.get_shape() == (conf.batch_size, 32*6*6, 8)

        # Reshape input
        u = tf.reshape(inputs, [conf.batch_size, 32*6*6, 1, 8, 1])
        u = tf.tile(u, [1, 1, num_caps, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 1]
        
        # Dynamic routing
        bij = tf.zeros((32*6*6, num_caps), name='b')
        wij = tf.get_variable('wij', shape=(1, 32*6*6, num_caps, 8, 16), initializer=tf.contrib.layers.xavier_initializer())
        w = tf.tile(wij, [conf.batch_size, 1, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 16]
        # bij = tf.tile(tf.zeros((32, num_caps), name='b'), [6*6, 1])  # [32*6*6, num_caps]
        # wij = tf.get_variable('wij', shape=(1, 32, num_caps, 8, 16))
        # w = tf.tile(wij, [conf.batch_size, 6*6, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 8, 16]

        # uhat
        uhat = tf.matmul(u, w, transpose_a=True)  # [batch_size, 32*6*6, num_caps, 1, 16]
        uhat = tf.reshape(uhat, [conf.batch_size, 32*6*6, num_caps, 16])  # [batch_size, 32*6*6, num_caps, 16]
        uhat_stop_grad = tf.stop_gradient(uhat)
        assert uhat.get_shape() == (conf.batch_size, 32*6*6, num_caps, 16)

        for r in range(num_iters):
            with tf.variable_scope('routing_iter_' + str(r)):
                # cij
                cij = tf.nn.softmax(bij, dim=-1)  # [32*6*6, num_caps]
                cij = tf.tile(tf.reshape(cij, [1, 32*6*6, num_caps, 1]),
                              [conf.batch_size, 1, 1, 1])  # [batch_size, 32*6*6, num_caps, 1]
                assert cij.get_shape() == (conf.batch_size, 32*6*6, num_caps, 1)

                if r == num_iters-1: 
                    # s, v
                    s = tf.reduce_sum(tf.multiply(uhat, cij), axis=1)  # [batch_size, num_caps, 16]
                    v = self.squash(s)  # [batch_size, num_caps, 16]
                    assert v.get_shape() == (conf.batch_size, num_caps, 16)
                else: 
                    # s, v (with no gradient propagation)
                    s = tf.reduce_sum(tf.multiply(uhat_stop_grad, cij), axis=1)  # [batch_size, num_caps, 16]
                    v = self.squash(s)  # [batch_size, num_caps, 16]
                    assert v.get_shape() == (conf.batch_size, num_caps, 16)
                    
                    # update b
                    vr = tf.reshape(v, [conf.batch_size, 1, num_caps, 16])
                    a = tf.reduce_sum(tf.reduce_sum(tf.multiply(uhat_stop_grad, vr), axis=0), axis=2)  # [32*6*6, num_caps]
                    bij = bij + a
                    assert a.get_shape() == (32*6*6, num_caps)
        return v

    def squash(self, s):
        s_l2 = tf.sqrt(tf.reduce_sum(tf.square(s), axis=-1, keep_dims=True) + conf.eps)
        scalar_factor = tf.square(s_l2) / (1 + tf.square(s_l2))
        v = scalar_factor * tf.divide(s, s_l2)
        return v

    def reconstruction(self, inputs, targets):
        # Validation
        assert inputs.get_shape() == (conf.batch_size, 10, 16)
        assert targets.get_shape() == (conf.batch_size)

        with tf.variable_scope('masking'):
            enum = tf.cast(tf.range(conf.batch_size), dtype=tf.int64)
            enum_indices = tf.concat(
                [tf.expand_dims(enum, 1), tf.expand_dims(targets, 1)],
                axis=1
            )
            assert enum_indices.get_shape() == (conf.batch_size, 2)

            masked_inputs = tf.gather_nd(inputs, enum_indices)
            assert masked_inputs.get_shape() == (conf.batch_size, 16)

        with tf.variable_scope('reconstruction'):
            fc_relu1 = tf.contrib.layers.fully_connected(
                inputs=masked_inputs,
                num_outputs=512,
                activation_fn=tf.nn.relu
            )
            # fc_relu1 = tf.nn.dropout(fc_relu1, keep_prob=0.9)
            fc_relu2 = tf.contrib.layers.fully_connected(
                inputs=fc_relu1,
                num_outputs=1024,
                activation_fn=tf.nn.relu
            )
            # fc_relu2 = tf.nn.dropout(fc_relu2, keep_prob=0.9)
            fc_sigmoid = tf.contrib.layers.fully_connected(
                inputs=fc_relu2,
                num_outputs=784,
                activation_fn=tf.nn.sigmoid
            )
            assert fc_sigmoid.get_shape() == (conf.batch_size, 784)
            recons = tf.reshape(fc_sigmoid, shape=(conf.batch_size, 28, 28))

        return recons

    def margin_loss(self, logits, labels, mplus=0.9, mminus=0.1, lambd=0.5):
        left = tf.square(tf.maximum(0., mplus - logits))
        right = tf.square(tf.maximum(0., logits - mminus))
        assert left.get_shape() == (conf.batch_size, 10)
        assert right.get_shape() == (conf.batch_size, 10)

        T_k = labels
        L_k = T_k * left + lambd * (1-T_k) * right
        mloss = tf.reduce_mean(tf.reduce_sum(L_k, axis=1))
        return mloss

    def reconstruction_loss(self, origin, decoded):
        origin = tf.reshape(origin, shape=(conf.batch_size, -1))
        decoded = tf.reshape(decoded, shape=(conf.batch_size, -1))
        rloss = tf.reduce_mean(tf.square(decoded - origin))
        return rloss

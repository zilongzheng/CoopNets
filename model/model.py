from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from six.moves import xrange

from model.utils.interpolate import *
from model.utils.custom_ops import *
from model.utils.data_io import DataSet, saveSampleResults
from model.utils.vis_util import Visualizer


class CoopNets(object):
    def __init__(self, num_epochs=200, image_size=64, batch_size=100, nTileRow=12, nTileCol=12, net_type='object',
                 d_lr=0.001, g_lr=0.0001, beta1=0.5,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016,
                 gen_step_size=0.1, gen_sample_steps=0, gen_refsig=0.3,
                 data_path='/tmp/data/', log_step=10, category='rock',
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test'):
        self.type = net_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nTileRow = nTileRow
        self.nTileCol = nTileCol
        self.num_chain = nTileRow * nTileCol
        self.beta1 = beta1

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.delta1 = des_step_size
        self.sigma1 = des_refsig
        self.delta2 = gen_step_size
        self.sigma2 = gen_refsig
        self.t1 = des_sample_steps
        self.t2 = gen_sample_steps

        self.data_path = os.path.join(data_path, category)
        self.log_step = log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.test_dir = test_dir

        if self.type == 'texture':
            self.z_size = 49
        elif self.type == 'object':
            self.z_size = 100
        elif self.type == 'object_small':
            self.z_size = 2

        self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs')
        self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z')

        self.debug = False

    def build_model(self):
        self.gen_res = self.generator(self.z, reuse=False)

        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)

        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

        self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0)))

        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1)
        des_grads_vars = des_optim.compute_gradients(self.des_loss, var_list=des_vars)
        # des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)

        # generator variables
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]

        self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res),
                                       axis=0))

        gen_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1)
        gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
        # gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
        self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevins
        self.langevin_descriptor = self.langevin_dynamics_descriptor(self.syn)
        self.langevin_generator = self.langevin_dynamics_generator(self.z)

        tf.summary.scalar('des_loss', self.des_loss)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('recon_err', self.recon_err)

        self.summary_op = tf.summary.merge_all()

    def langevin_dynamics_descriptor(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.t1)

        def body(i, syn):
            noise = tf.random_normal(shape=[self.num_chain, self.image_size, self.image_size, 3], name='noise')
            syn_res = self.descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad) + self.delta1 * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def langevin_dynamics_generator(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.t2)

        def body(i, z):
            noise = tf.random_normal(shape=[self.num_chain, self.z_size], name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - gen_res),
                                       axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def train(self, sess):
        self.build_model()

        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        num_batches = int(math.ceil(len(train_data) / self.batch_size))

        # initialize training
        sess.run(tf.global_variables_initializer())

        sample_results = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, 3)

        saver = tf.train.Saver(max_to_keep=50)

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        with open(self.model_dir + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))

        des_loss_vis = Visualizer(title='descriptor', ylabel='normalized negative log-likelihood', ylim=(-200, 200),
                                  save_figpath=self.log_dir + '/des_loss.png', avg_period = self.batch_size)

        gen_loss_vis = Visualizer(title='generator', ylabel='reconstruction error',
                                  save_figpath=self.log_dir + '/gen_loss.png', avg_period = self.batch_size)


        # train
        for epoch in xrange(self.num_epochs):
            start_time = time.time()
            des_loss_avg, gen_loss_avg, mse_avg = [], [], []
            for i in xrange(num_batches):

                obs_data = train_data[i * self.batch_size:min(len(train_data), (i + 1) * self.batch_size)]

                # Step G0: generate X ~ N(0, 1)
                z_vec = np.random.randn(self.num_chain, self.z_size)
                g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
                # Step D1: obtain synthesized images Y
                if self.t1 > 0:
                    syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})
                # Step G1: update X using Y as training image
                if self.t2 > 0:
                    z_vec = sess.run(self.langevin_generator, feed_dict={self.z: z_vec, self.obs: syn})
                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.apply_d_grads],
                                  feed_dict={self.obs: obs_data, self.syn: syn})[0]
                # Step G2: update G net
                g_loss = sess.run([self.gen_loss, self.apply_g_grads],
                                  feed_dict={self.obs: syn, self.z: z_vec})[0]

                # Compute MSE for generator
                mse = sess.run(self.recon_err, feed_dict={self.obs: syn, self.syn: g_res})
                sample_results[i * self.num_chain:(i + 1) * self.num_chain] = syn

                des_loss_avg.append(d_loss)
                gen_loss_avg.append(g_loss)
                mse_avg.append(mse)

                des_loss_vis.add_loss_val(epoch*num_batches + i, d_loss / float(self.image_size * self.image_size * 3))
                gen_loss_vis.add_loss_val(epoch*num_batches + i, mse)

                if self.debug:
                    print('Epoch #{:d}, [{:2d}]/[{:2d}], descriptor loss: {:.4f}, generator loss: {:.4f}, '
                          'L2 distance: {:4.4f}'.format(epoch, i + 1, num_batches, d_loss.mean(), g_loss.mean(), mse))
                if i == 0 and epoch % self.log_step == 0:
                    if not os.path.exists(self.sample_dir):
                        os.makedirs(self.sample_dir)
                    saveSampleResults(syn, "%s/des%03d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)
                    saveSampleResults(g_res, "%s/gen%03d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)

            end_time = time.time()
            print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
                  'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), end_time - start_time))

            if epoch % self.log_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

                des_loss_vis.draw_figure()
                gen_loss_vis.draw_figure()


    def test(self, sess, ckpt, sample_size):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.z, reuse=False)

        num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        for i in xrange(num_batches):
            z_vec = np.random.randn(min(sample_size, self.num_chain), self.z_size)
            g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            saveSampleResults(g_res, "%s/gen%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

            # output interpolation results
            interp_z = linear_interpolator(z_vec, npairs=self.nTileRow, ninterp=self.nTileCol)
            interp = sess.run(gen_res, feed_dict={self.z: interp_z})
            saveSampleResults(interp, "%s/interp%03d.png" % (self.test_dir, i), col_num=self.nTileCol)
            sample_size = sample_size - self.num_chain

    def descriptor(self, inputs, reuse=False):
        with tf.variable_scope('des', reuse=reuse):
            if self.type == 'object':
                conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv1")

                conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv2")

                conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_fn=leaky_relu,
                               name="conv3")

                fc = fully_connected(conv3, 100, name="fc")

                return fc
            else:
                return NotImplementedError

    def generator(self, inputs, reuse=False, is_training=True):
        with tf.variable_scope('gen', reuse=reuse):
            if self.type == 'object':
                inputs = tf.reshape(inputs, [-1, 1, 1, self.z_size])
                convt1 = convt2d(inputs, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4)
                                 , strides=(1, 1), padding="VALID", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = leaky_relu(convt1)

                convt2 = convt2d(convt1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = leaky_relu(convt2)

                convt3 = convt2d(convt2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = leaky_relu(convt3)

                convt4 = convt2d(convt3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = leaky_relu(convt4)

                convt5 = convt2d(convt4, (None, self.image_size, self.image_size, 3), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt5")
                convt5 = tf.nn.tanh(convt5)

                return convt5
            else:
                return NotImplementedError

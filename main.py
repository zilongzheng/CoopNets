from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.model import CoopNet

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('image_size', 64, 'Image size to rescale images')
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 250, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 12, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 12, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.01, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 10, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0001, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')

tf.flags.DEFINE_string('data_dir', './data', 'The data directory')
tf.flags.DEFINE_string('category', 'rock_arch_small', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 100, 'Number of images to generate during test.')


def main(_):
    model = CoopNet(
        net_type='object',
        num_epochs=FLAGS.num_epochs,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        beta1=FLAGS.beta1,
        nTileRow=FLAGS.nTileRow, nTileCol=FLAGS.nTileCol,
        d_lr=FLAGS.d_lr, g_lr=FLAGS.g_lr,
        des_refsig=FLAGS.des_refsig, gen_refsig=FLAGS.gen_refsig,
        des_step_size=FLAGS.des_step_size, gen_step_size=FLAGS.gen_step_size,
        des_sample_steps=FLAGS.des_sample_steps, gen_sample_steps=FLAGS.gen_sample_steps,
        log_step=FLAGS.log_step, data_path=FLAGS.data_dir, category=FLAGS.category, output_dir=FLAGS.output_dir
    )

    with tf.Session() as sess:
        if FLAGS.test:
            model.test(sess, FLAGS.ckpt, FLAGS.sample_size)
        else:
            model.train(sess)


if __name__ == '__main__':
    tf.app.run()

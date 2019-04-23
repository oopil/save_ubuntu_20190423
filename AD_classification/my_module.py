from __future__ import division
import tensorflow as tf
from my_ops import *
from my_utils import *

def AE(image, options, train_AE=True, name='autoencoder'):

    if train_AE:
        pass
    else:
        pass


    pass


def disc_my(image, options, reuse=False, name="discriminator"):
    shape_print = False
    print('discriminating ...')
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        if shape_print:
            print(image.shape)
        h0 = lrelu(conv3d(image, options.df_dim,  ks=4, s=(2,2,2),name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv3d(h0, options.df_dim*2,  ks=4, s=(2,2,2),name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(deconv3d(h1, options.df_dim*2, ks=4, s=(2,2,2), name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(deconv3d(h2, options.df_dim*2,  ks=4, s=(2,2,2),name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = deconv3d(h3, 1, ks=1, s=(1,1,1), name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

def disc(image, options, reuse=False, name="discriminator"):
    shape_print = False
    print('discriminating ...')
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        if shape_print:
            print(image.shape)
        h0 = lrelu(conv3d(image, options.df_dim,  ks=4, s=(2,2,2),name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv3d(h0, options.df_dim*2,  ks=4, s=(2,2,2),name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(deconv3d(h1, options.df_dim*2, ks=4, s=(2,2,2), name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(deconv3d(h2, options.df_dim*2,  ks=4, s=(2,2,2),name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = deconv3d(h3, 1, ks=1, s=(1,1,1), name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

def generator_res(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def self_attention(x, dim, ks=4, s=2, name='attention'):
            pass

        def residule_block(x, dim, ks=3, s=(1,1,1), name='res'):
            # print('residual block input shape : {}'.format(x.shape))
            # p = int((ks - 1) / 2)
            # y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv3d(x, dim, ks, s, padding='SAME', name=name+'_c1'), name+'_bn1')
            # y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv3d(y, dim, ks, s, padding='SAME', name=name+'_c2'), name+'_bn2')
            # print('residual block y shape : {}'.format(y.shape))
            return y + x
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        # c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # print('imput image shape : {}'.format(image.shape))
        c1 = tf.nn.relu(instance_norm(conv3d(image, options.gf_dim,name='g_e1_c'), 'g_e1_bn'))
        # print('c1 shape : {}'.format(c1.shape))
        c2 = tf.nn.relu(instance_norm(conv3d(c1, options.gf_dim*2, name='g_e2_c'), 'g_e2_bn'))
        # print('c2 shape : {}'.format(c2.shape))
        c3 = tf.nn.relu(instance_norm(conv3d(c2, options.gf_dim*4,name='g_e3_c'), 'g_e3_bn'))
        # print('c3 shape : {}'.format(c3.shape))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        d1 = deconv3d(r6, options.gf_dim*2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv3d(d1, options.gf_dim, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        # d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(deconv3d(d1, options.output_c_dim, name='g_pred_c'))
        # print('resnet outupt shape : {}'.format(pred.shape))
        return pred

def generator_res_2(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        def residule_block(x, dim, ks=3, s=(1,1,1), name='res'):
            # print('residual block input shape : {}'.format(x.shape))
            y = instance_norm(conv3d(x, dim, ks, s, padding='SAME', name=name+'_c1'), name+'_bn1')
            y = instance_norm(conv3d(y, dim, ks, s, padding='SAME', name=name+'_c2'), name+'_bn2')
            return y + x
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # print('imput image shape : {}'.format(image.shape))
        c1 = tf.nn.relu(instance_norm(conv3d(image, options.gf_dim, ks=4, s=(2,2,2),name='g_e1_c'), 'g_e1_bn'))
        # print('c1 shape : {}'.format(c1.shape))
        c2 = tf.nn.relu(instance_norm(conv3d(c1, options.gf_dim*2, ks=4, s=(2,2,2), name='g_e2_c'), 'g_e2_bn'))
        # print('c2 shape : {}'.format(c2.shape))
        c3 = tf.nn.relu(instance_norm(conv3d(c2, options.gf_dim*4, ks=4, s=(2,2,2) ,name='g_e3_c'), 'g_e3_bn'))
        # print('c3 shape : {}'.format(c3.shape))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        d1 = deconv3d(r4, options.gf_dim*2, ks=4, s=(2,2,2), name='g_d1_dc')
        d1 = tf.nn.relu(d1)
        d2 = deconv3d(d1, options.gf_dim, ks=4, s=(2,2,2), name='g_d2_dc')
        d2 = tf.nn.relu(d2)
        # d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = deconv3d(d2, options.output_c_dim, ks=4, s=(2,2,2), name='g_pred_c')
        # pred = tf.nn.tanh(deconv3d(d1, options.output_c_dim, name='g_pred_c'))
        # print('resnet outupt shape : {}'.format(pred.shape))
        return pred

def generator_res_3(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        def residule_block(x, dim, ks=3, s=(1,1,1), name='res'):
            # print('residual block input shape : {}'.format(x.shape))
            y = conv3d(x, dim, ks, s, padding='SAME', name=name+'_c1')
            y = conv3d(y, dim, ks, s, padding='SAME', name=name+'_c2')
            return y + x
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # print('imput image shape : {}'.format(image.shape))
        c1 = tf.nn.relu(conv3d(image, options.gf_dim, ks=4, s=(2,2,2),name='g_e1_c'))
        # print('c1 shape : {}'.format(c1.shape))
        c2 = tf.nn.relu(conv3d(c1, options.gf_dim*2, ks=4, s=(2,2,2), name='g_e2_c'))
        # print('c2 shape : {}'.format(c2.shape))
        c3 = tf.nn.relu(conv3d(c2, options.gf_dim*4, ks=4, s=(2,2,2) ,name='g_e3_c'))
        # print('c3 shape : {}'.format(c3.shape))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        d1 = deconv3d(r9, options.gf_dim*2, ks=4, s=(2,2,2), name='g_d1_dc')
        d1 = tf.nn.relu(d1)
        d2 = deconv3d(d1, options.gf_dim, ks=4, s=(2,2,2), name='g_d2_dc')
        d2 = tf.nn.relu(d2)
        # d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = deconv3d(d2, options.output_c_dim, ks=4, s=(2,2,2), name='g_pred_c')
        # pred = tf.nn.tanh(deconv3d(d1, options.output_c_dim, name='g_pred_c'))
        # print('resnet outupt shape : {}'.format(pred.shape))
        return pred

def generator_test(image, options, reuse = False, name="generator"):
    print('generating ...')
    shape_print = True
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        print('options gf dim : ',options.gf_dim)
        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv3d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv3d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv3d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # print('e3 shape : ', e3.shape)
        # e3 is (32 x 32 x self.gf_dim*4)
        d1 = deconv3d(tf.nn.relu(e3), options.gf_dim*2, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e2], 4)
        # print('d1 shape : ', d1.shape)
        # d1 is (64 x 64 x self.gf_dim*8*2)

        d2 = deconv3d(tf.nn.relu(d1), options.gf_dim*1, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e1], 4)
        # d2 is (128 x 128 x self.gf_dim*8*2)
        d3 = deconv3d(tf.nn.relu(d2), options.output_c_dim, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        # d3 is (256 x 256 x self.gf_dim*8*2)
        if shape_print:
            print(image.shape, e1.shape, e2.shape, e3.shape, d1.shape, d2.shape, d3.shape)
        return tf.nn.tanh(d3)

def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv3d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv3d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv3d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv3d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv3d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv3d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv3d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv3d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv3d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv3d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv3d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv3d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv3d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv3d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv3d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv3d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

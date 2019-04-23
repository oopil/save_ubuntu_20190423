"""
the code I refer to is the code below.
I developed the simple code into 8 options prof. indicated

CIFAR-10 Convolutional Neural Networks(CNN) Example
next_batch function is copied from edo's answer
https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
Author : solaris33
Project URL : http://solarisailab.com/archives/2325
"""
#import torch
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
# read next batch function
def next_batch(num, data, labels):
    #return samples as much as num
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def relu(input):
    return tf.nn.relu(input)

def batch_normalization(input):
    return tf.layers.batch_normalization(input)

def one_cnn_layer(input, var_name, input_ch, output_ch):
    W_conv_1 = tf.get_variable(var_name + str(1), shape=[1,1,input_ch,input_ch/2], initializer=tf.contrib.layers.xavier_initializer())
    output = tf.nn.conv2d(input, W_conv_1, strides=[1, 1, 1, 1], padding='SAME')
    W_conv_2 = tf.get_variable(var_name + str(2), shape=[3,3,input_ch/2,input_ch/2], initializer=tf.contrib.layers.xavier_initializer())
    output = tf.nn.conv2d(output, W_conv_2, strides=[1, 1, 1, 1], padding='SAME')
    W_conv_3 = tf.get_variable(var_name + str(3), shape=[3,3,input_ch/2,output_ch], initializer=tf.contrib.layers.xavier_initializer())
    output = tf.nn.conv2d(output, W_conv_3, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(output)

def wide_residual_net(input, kernel_num, var_name, input_ch, output_ch):
    output = input
    for num in range(1, kernel_num + 1):
        output = output + one_cnn_layer(input, var_name + str(num), input_ch, output_ch)
    return output

def build_model(x):
    #input image &
    #Batch normalization
    x_image = relu(batch_normalization(x))
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], stddev=5e-2))
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], stddev=5e-2))
    #W_conv1 = tf.get_variable("W-conv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    #W_conv2 = tf.get_variable("W-conv2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    #W_conv3 = tf.get_variable("W-conv3", shape=[5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    #W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 128], stddev=5e-2))
    #W_conv1 = tf.get_variable('W_conv1',shape=[5, 5, 3, 128], initializer=tf.keras.initializer.he_normal(seed=None))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv1 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    batch_relu_1 = relu(batch_normalization((h_conv1)))

    net_width = 2
    h_conv2 = wide_residual_net(batch_relu_1, net_width, 'wide_net_1', 128, 128)
    batch_relu_2 = relu(batch_normalization(h_conv2))
    h_conv2 = wide_residual_net(batch_relu_1, net_width, 'wide_net_2', 128, 128)
    batch_relu_2 = relu(batch_normalization(h_conv2))
    h_conv2 = wide_residual_net(batch_relu_1, net_width, 'wide_net_3', 128, 128)
    batch_relu_2 = relu(batch_normalization(h_conv2))
    h_conv2 = wide_residual_net(batch_relu_1, net_width, 'wide_net_4', 128, 128)
    batch_relu_2 = relu(batch_normalization(h_conv2))
    h_conv2 = wide_residual_net(batch_relu_1, net_width, 'wide_net_5', 128, 128)
    batch_relu_2 = relu(batch_normalization(h_conv2))
    h_conv4 = wide_residual_net(batch_relu_2, net_width, 'wide_net_6', 128, 128)
    # Fully connected layer

    last_image_size = 8
    W_fc1 = tf.get_variable("W-fc1",shape=[last_image_size * last_image_size * 128, 384], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_conv4_flat = tf.reshape(h_conv4, [-1, last_image_size * last_image_size * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.get_variable("W-fc2",shape=[384, 10],initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

#define place holders
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
#load datasets - use keras lib
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#one hot encoding the label
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
print(x_train.shape)
print(y_train_one_hot, y_test_one_hot)
print(tf.one_hot(y_train,10) )

y_pred, logits = build_model(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# learning rate decaying part
step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(1e-3, step, 1, 0.9999)
train_step = tf.train.AdamOptimizer(rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.01, momentum=0.9).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
restore = False
with tf.Session() as sess:
    #initialization
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.contrib.layers.xavier_initializer())
    epoch_num = 2000
    for epoch in range(epoch_num):
        batch = next_batch(64, x_train, y_train_one_hot.eval())

        if epoch % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            print("Epoch: %d, training accuracy: %f, loss: %f" % (epoch, train_accuracy, loss_print))
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print('model is saved in path : %s' %save_path)

    # if you want restore the parameters, load it
    if restore == True:
        saver.restore(sess, 'tmp/model.ckpt')
        print('model is restored.')

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("test data accuracy: %f" % test_accuracy)
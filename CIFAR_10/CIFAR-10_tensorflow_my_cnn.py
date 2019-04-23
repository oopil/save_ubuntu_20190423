"""
CIFAR-10 Convolutional Neural Networks(CNN) Example
next_batch function is copied from edo's answer
https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
Author : solaris33
Project URL : http://solarisailab.com/archives/2325
reference :
https://github.com/solaris33/dl_cv_tensorflow_10weeks/blob/master/week2/cifar10_classification_using_cnn.py
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
    W_conv_3 = tf.get_variable(var_name + str(3), shape=[1,1,input_ch/2,output_ch], initializer=tf.contrib.layers.xavier_initializer())
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
    # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping)합니다.
    W_conv1 = tf.get_variable("W-conv1", shape=[5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
    #W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    #W_conv1 = tf.get_variable('W_conv1',shape=[5, 5, 3, 64], initializer=tf.keras.initializer.he_normal(seed=None))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME'))
    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    batch_relu_1 = relu(batch_normalization((h_pool1)))
    # 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
    h_conv2 = wide_residual_net(batch_relu_1, 3, 'wide_net_1', 64, 64)
    batch_relu_2 = relu(batch_normalization(h_conv2))

    W_conv3 = tf.get_variable("W-conv3", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer())
    h_conv3 = tf.nn.relu(tf.nn.conv2d(batch_relu_2, W_conv3, strides=[1, 1, 1, 1], padding='SAME'))
    batch_relu_3 = relu(batch_normalization(h_conv3))
    # 네번째 convolutional layer
    #W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    W_conv4 = tf.get_variable("W-conv4", shape=[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer())
    h_conv4 = tf.nn.relu(tf.nn.conv2d(batch_relu_3, W_conv4, strides=[1, 1, 1, 1], padding='SAME'))
    # Fully connected layer
    W_fc1 = tf.get_variable("W-fc1",shape=[16 * 16 * 128, 384], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_conv4_flat = tf.reshape(h_conv4, [-1, 16 * 16 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
    #W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
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
# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
#train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
train_step = tf.train.MomentumOptimizer(0.01, momentum=0.9).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
restore = False
with tf.Session() as sess:
    #initialization
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.contrib.layers.xavier_initializer())

    epoch_num = 300
    for epoch in range(epoch_num):
        batch = next_batch(64, x_train, y_train_one_hot.eval())

        # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if epoch % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            print("Epoch: %d, training accuracy: %f, loss: %f" % (epoch, train_accuracy, loss_print))
        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

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
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:16:05 2018

@author: xim11
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import os
import cv2
import argparse
import tempfile
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#%%
case_num = 4
#Accuracy = [[] for i in range(case_num)]
Accuracy = []
Accuracy_per = []
Try_count = 0
layer_filter_constant = [3]
filter_num = len(layer_filter_constant)
Accuracy_case_per = [[] for _ in range(case_num)]


#%%

def deepnn(x, filter_size):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
#    with tf.name_scope('reshape'):
#        x_image = tf.reshape(x, [-1, 60, 60, 1])
    layer_filter_size = [filter_size, filter_size]
    filter_size_1 = layer_filter_size[0]
    filter_size_2 = layer_filter_size[1]
    
    # x is 60 by 60 image data
    x_image = tf.reshape(x, [-1,60, 60, 1])
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([filter_size_1, filter_size_1, 1, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([filter_size_2, filter_size_2, 32, 64], 'W_conv2')
        b_conv2 = bias_variable([64], 'b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([15 * 15 * 64, 1024], 'W_fc1')
        b_fc1 = bias_variable([1024], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 4], 'W_fc1')
        b_fc2 = bias_variable([4], 'b_fc2')
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    return y_conv, keep_prob
#%%

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, Name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = Name)

def bias_variable(shape, Name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = Name)


def LoadImageFolder(data_whole, data_path, case):
    file_name_list = os.listdir(data_path)
    file_name_list.sort()
    image_num = len(file_name_list)
    print("%d images are found." %(image_num))
    item_num = len(file_name_list)
    
    for i in range(item_num):
        image_path = os.path.join(data_path, file_name_list[i])
        new_image = cv2.imread(image_path,0)
        data_whole[case].append(new_image)
#        label = np.zeros(4)
#        label[case] = 1
#        print(case, label)

def GetCase(data_path):
    if data_path[-1] == 'A':
        case = 0
    elif data_path[-1] == 'B':
        case = 1
    elif data_path[-1] == 'C':
        case = 2
    elif data_path[-1] == 'D':
        case = 3
    return case
    
def LoadImages(data_whole):
    folder_path = 'spine_cut'
    folder_name_list = os.listdir(folder_path)
    folder_name_list.sort()
    
    #load images
    for folder_name in folder_name_list:
        data_path = os.path.join(folder_path, folder_name)
        case = GetCase(data_path)
        LoadImageFolder(data_whole, data_path, case)

#    train_image_num = len(image_train_data)
#    test_image_num = len(image_test_data)
#    print('train image number is ' , train_image_num)
#    print('test image number is ' , test_image_num)
    
def cut_image(x, y, w, h, img):
    interval_row = int(w/2)
    interval_col = int(h/2)
    img_trim = img[x-interval_row:x+interval_row, y-interval_col:y+interval_col]
    return img_trim
#
#def Hold_limits(array, low_boundary, high_boundary):
#    rows , cols = np.shape(array)
#    for row in array
#%%
    
def PrintAffine(image, dst, affine_name):
    print('<<{} transform>>'.format(affine_name))
    plt.figure(figsize = (4,2))
    plt.subplot(121),plt.imshow(image, cmap = 'gray'),plt.title('Input')
    plt.subplot(122),plt.imshow(dst, cmap = 'gray'),plt.title('Output')
    plt.show()
        
def AppendData(data_list, label_list, data_case_list, dst, label):
    data_list.append(dst)
    data_case_list.append(dst)
    label_list.append(label)

def AffineAndAppend(data_list, label_list, data_case_list, image, label):
    #affine transformation and increase data set
    rows, cols = image.shape
    
    #horizontal flip
    dst = cv2.flip(image, 1)
    AppendData(data_list, label_list, data_case_list, dst, label)
    
def AugmentWithRandomNoise(image, intensity):
    rows, cols = image.shape
    tmp = image + np.random.randn(rows, cols) * intensity
    
#    Gaussian = np.random.normal(0,1, 3600)
#    tmp = np.reshape(Gaussian,(rows,cols))
    new_image = image + tmp * intensity
    return new_image

def AugmentAndAppend(data_list, label_list, data_case_list, image, label):
    
    #affine transformation and increase data set
    rows, cols = image.shape
    intensity = [3*(i+1) for i in range(7)]
    
    #change intensity with normal distribution or contant filter
    for n in range(len(intensity)):
        dst = AugmentWithRandomNoise(image, intensity[n])
        AppendData(data_list, label_list, data_case_list, dst, label)
    
#    PrintAffine(image, dst, 'change intensity with normal distribution')
    
    tmp = image*1.5
    x, y = tmp.shape
    dst = tmp
    AppendData(data_list, label_list, data_case_list, dst, label)
#    PrintAffine(image, dst, 'change intensity brighter with contant filter')
    
    tmp = image*0.5
    x, y = tmp.shape
    dst = cut_image(int(x/2), int(y/2), rows, cols, tmp)
    AppendData(data_list, label_list, data_case_list, dst, label)
#    PrintAffine(image, dst, 'change intensity darker with contant filter')
    
def PartialThreshing(image_data, low_intensity_boundary):
    rows, cols = image_data.shape
#    print(rows, cols)
    for row in range(rows):
        for col in range(cols):
            if image_data[row,col] < low_intensity_boundary:
                image_data[row,col] = 0
        
def ClassifyCase(data_whole, class_constant):
    data_train = []
    data_test = []
    tmp_label_train = []
    tmp_label_test = []
    data_train_case = [[] for i in range(4)]
    data_test_case = [[] for i in range(4)]

    for i in range(4):
        count = class_constant
        #affine transformation and increase data set
        for patch_data in data_whole[i]:
            #try threshold image learning
            data = patch_data
#            ret, data = cv2.threshold(patch_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#            PartialThreshing(patch_data, 100)
            
            if count % 4 == 0 :
                data_test.append(data)
                data_test_case[i].append(data)
                tmp_label_test.append(i)
#                AffineAndAppend(data_test, tmp_label_test, data_test_case[i], data, i)
            else:
                data_train.append(data)
                data_train_case[i].append(data)
                tmp_label_train.append(i)
#                AffineAndAppend(data_train, tmp_label_train, data_train_case[i], data, i)
                AugmentAndAppend(data_train, tmp_label_train, data_train_case[i], data, i)
            count += 1
    
    #one hot encoding
    label_train = pd.get_dummies(tmp_label_train)
    label_test = pd.get_dummies(tmp_label_test)
    
#    print('<training label> \n', label_train)
#    print('<testing label> \n', label_test)
    print('train data count is : ', len(data_train))
    print('test data count is : ', len(data_test))

    return data_train, data_test, label_train, label_test, data_train_case, data_test_case

#def ShowWrongAnswer(data_case):
    
def GetCaseByIndex(index):
    if index == 0:
        return 'A'
    elif index == 1:
        return 'B'
    elif index == 2:
        return 'C'
    elif index == 3:
        return 'D'
            
#%%

def main(class_constant, filter_num):
    # Import data
    case_num = 4
    data_whole = [[] for i in range(case_num)]
    LoadImages(data_whole)
    print(np.shape(data_whole[0]))
    data_train, data_test, label_train, label_test, data_train_case, data_test_case = ClassifyCase(data_whole, class_constant)
    
    print('training data is ready')
    print('test data is ready')
    
    count_test_data_case = []
    count_train_data_case = []
    
    for i in range(case_num):
        count_test_data_case.append(len(data_test_case[i]))
        count_train_data_case.append(len(data_train_case[i]))
    
    label_test_case = [[] for i in range(case_num)]
    label_train_case = [[] for i in range(case_num)]
    for i in range(case_num):
        count_data_test = count_test_data_case[i]
        count_data_train = count_train_data_case[i]
        
        for _ in range(count_data_test):
            label = np.zeros(4)
            label[i] = 1
            label_test_case[i].append(label)  
            
        for _ in range(count_data_train):
            label = np.zeros(4)
            label[i] = 1
            label_train_case[i].append(label)      
    
    print('each case training count is ', count_train_data_case)
    print('each case testing count is ', count_test_data_case)
    print('one hot encoding is finished')
    
    #stop intentionally
#    assert False
    
    # Create the modelimg
    with tf.name_scope("input") as scope:
        x = tf.placeholder(tf.float32, [None, 60, 60], name = "input")

    # Define loss and optimizer
    with tf.name_scope("y_") as scope:
        y_ = tf.placeholder(tf.float32, [None, 4], name = "y_")

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x, layer_filter_constant[filter_num])

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    
    accuracy = tf.reduce_mean(correct_prediction)
    prediction = [[] for _ in range(4)]
    
#    save_path = "data_v2/"
#    os.mkdir( save_path );
    
    epoch = 1    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
         
        for i in range(epoch):
            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: data_train, y_: label_train, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: data_train, y_: label_train, keep_prob: 0.5})
        
        accuracy_tmp = accuracy.eval(feed_dict={ x: data_test, y_: label_test, keep_prob: 1.0})
        print('filter {} test accuracy {}' .format(filter_num,accuracy_tmp))
        Accuracy.append('filter {} test accuracy {}' .format(filter_num,accuracy_tmp))
        Accuracy_per.append(accuracy_tmp)
        
        for i in range(case_num):
            test_accuracy = accuracy.eval(feed_dict={ x: data_test_case[i], y_: label_test_case[i], keep_prob: 1.0})
            print('test accuracy case %d : %g (testing count %d)' % (i, test_accuracy, count_test_data_case[i]))
            Accuracy_case_per[i].append(test_accuracy)
            Accuracy.append('test accuracy case %d : %g (testing count %d)' % (i, test_accuracy, count_test_data_case[i]))
    
def test_for_filter_size(filter_num):
    for class_constant in range(4):
        main(class_constant, filter_num)

for filter in range(filter_num):
    test_for_filter_size(filter)
    
print('\n<<<<<Total result >>>>>\n')
for accuracy_info in Accuracy:
    print(accuracy_info)
for n in range(filter_num):
    print("filter num {}".format(n))
    print('\naverage accuracy is : ',np.average(Accuracy_per[n]))
    index = 4*n
    for case in range(case_num):
        print('case {} average accuracy : {} => {}'.format(GetCaseByIndex(case), Accuracy_case_per[case][n], np.average(Accuracy_case_per[case][index:index+4])))
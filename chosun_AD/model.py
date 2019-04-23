#importing dependencies
import tensorflow as tf
import pandas as pd
from data import *

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#Helper functions to define weights and biases
def init_weights(shape):
    '''
    Input: shape -  this is the shape of a matrix used to represent weigts for the arbitrary layer
    Output: wights randomly generated with size = shape
    '''
    return tf.Variable(tf.truncated_normal(shape, 0, 0.05))

def init_biases(shape):
    '''
    Input: shape -  this is the shape of a vector used to represent biases for the arbitrary layer
    Output: a vector for biases (all zeros) lenght = shape
    '''
    return tf.Variable(tf.zeros(shape))


def fully_connected_layer(inputs, input_shape, output_shape, keep_prob, activation=tf.nn.relu):
    '''
    This function is used to create tensorflow fully connected layer.

    Inputs: inputs - input data to the layer
            input_shape - shape of the inputs features (number of nodes from the previous layer)
            output_shape - shape of the layer
            activatin - used as an activation function for the layer (non-liniarity)
    Output: layer - tensorflow fully connected layer

    '''
    # definine weights and biases
    weights = init_weights([input_shape, output_shape])
    biases = init_biases([output_shape])
    # x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    # if activation argument is not None, we put layer values through an activation function
    if activation != None:
        layer = activation(layer)

    return layer

#%%
# splitting the dataset to the training set and the testing set
# hyper parameters
'''
when using the all 3 options to the features, 
I could observe high training speed and high testing accuracy.
'''
is_merge = True # True
option_num = 0 # P V T options
'''
I should set the class options like
NC vs AD
NC vs MCI
MCI vs AD

NC vs MCI vs AD
'''
class_option = ['NC vs AD','NC vs MCI','MCI vs AD','NC vs MCI vs AD']
class_option_index = 0
class_num = class_option_index//3 + 2

ford_num = 10
ford_index = 0
keep_prob = 0.9 # 0.7

learning_rate = 0.02
epochs = 2000
print_freq = 100
save_freq = 300
# batch_size = 50

data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)
# assert False
data, label = shuffle_two_arrays(data, label)
X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
# print(len(data[0]), len(X_train[0]))
X_train, Y_train = valence_class(X_train, Y_train, class_num)
X_test, Y_test = valence_class(X_test, Y_test, class_num)
feature_num = len(X_train[0])
print(X_train.shape, X_test.shape)
# assert False

graph = tf.Graph()
with graph.as_default():
    #Tensorflow placeholders - inputs to the TF graph
    inputs =  tf.placeholder(tf.float32, [None, feature_num], name='Inputs')
    targets =  tf.placeholder(tf.float32, [None, class_num], name='Targets')

    '''
    need to add normalization between each layers.
    '''
    # layer = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
    layer = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[1], keep_prob)# + l1
            l3 = fully_connected_layer(l2, layer[1], layer[2], keep_prob)# + l2

        with tf.name_scope("layer4"):
            l4 = fully_connected_layer(l3, layer[2], layer[3], keep_prob)# + l3
            l5 = fully_connected_layer(l4, layer[3], layer[4], keep_prob)# + l4# + l3
            l6 = fully_connected_layer(l5, layer[4], layer[5], keep_prob)# + l5# + l4 + l3
            l7 = fully_connected_layer(l6, layer[5], layer[6], keep_prob)# + l6# + l5 + l4 + l3
            l8 = fully_connected_layer(l7, layer[6], layer[7], keep_prob) + l4# + l7# + l6 + l5 + l4 + l3
        with tf.name_scope("layer5"):
            l9 = fully_connected_layer(l8, layer[2], layer[3], keep_prob)# + l8
            l10 = fully_connected_layer(l9, layer[3], layer[4], keep_prob)# + l9# + l8
            l11 = fully_connected_layer(l10, layer[4], layer[5], keep_prob)# + l10# + l9 + l8
            l12 = fully_connected_layer(l11, layer[5], layer[6], keep_prob)# + l11# + l10 + l9 + l8
            l13 = fully_connected_layer(l12, layer[6], layer[7], keep_prob) + l9# + l12# + l11 + l10 + l9 + l8
        with tf.name_scope("layer6"):
            l14 = fully_connected_layer(l13, layer[2], layer[3], keep_prob)# + l13
            l15 = fully_connected_layer(l14, layer[3], layer[4], keep_prob)# + l14# + l13
            l16 = fully_connected_layer(l15, layer[4], layer[5], keep_prob)# + l15# +l14 + l13
            l17 = fully_connected_layer(l16, layer[5], layer[6], keep_prob)# + l16# + l15 + l14 + l13
            l18 = fully_connected_layer(l17, layer[6], layer[7], keep_prob) + l14# + l17# + l16 + l15 + l14 + l13
        with tf.name_scope("layer7"):
            l19 = fully_connected_layer(l18, layer[2], layer[2], keep_prob)# + l18
            l20 = fully_connected_layer(l19, layer[3], layer[3], keep_prob)# + l19# + l18
            l21 = fully_connected_layer(l20, layer[4], layer[4], keep_prob)# + l20# + l19 + l18
            l22 = fully_connected_layer(l21, layer[5], layer[5], keep_prob)# + l21# + l20 + l19 + l18
            l23 = fully_connected_layer(l22, layer[6], layer[6], keep_prob) + l19# + l22# + l21 + l20 + l19 + l18
            l24 = fully_connected_layer(l23, layer[6], layer_last, keep_prob)# + l23
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

    #Mean_squared_error function and optimizer choice - Classical Gradient Descent
    cost = loss2 = tf.reduce_mean(tf.squared_difference(targets, predictions))
    tf.summary.scalar("cost", cost)
    merged_summary = tf.summary.merge_all()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # from tqdm import tqdm

    # Starting session for the graph
    top_train_accur = 0
    top_test_accur = 0
    train_accur_list = []
    test_accur_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())
        # writer = tf.train.SummaryWriter('./my_graph', graph=tf.get_default_graph())
        writer.add_graph(sess.graph)
        # TRAINING PORTION OF THE SESSION
        # one hot encoding
        Y_train = pd.get_dummies(Y_train)
        Y_train = np.array(Y_train)
        Y_test = pd.get_dummies(Y_test)
        Y_test = np.array(Y_test)
        for i in range(epochs):
            '''
            idx = np.random.choice(len(X_train), batch_size, replace=True)
            x_batch = X_train[idx, :]
            y_batch = Y_train[idx]
            y_batch = np.reshape(y_batch, (len(y_batch), 1))
            '''
            y_batch = Y_train
            x_batch = X_train

            summary, batch_loss, opt, preds_train = sess.run([merged_summary, cost, optimizer, predictions], \
                                                             feed_dict={inputs: x_batch, targets: y_batch})
            writer.add_summary(summary, global_step=i)
            train_accur = accuracy(preds_train, Y_train)
            # TESTING PORTION OF THE SESSION
            preds = sess.run([predictions], feed_dict={inputs: X_test})
            # preds_nparray = np.squeeze(np.array(preds), 0)
            preds_nparray = np.squeeze(np.array(preds), 0)
            test_accur = accuracy(preds_nparray, Y_test)

            if i % save_freq == 0:
                train_accur_list.append(train_accur//1)
                test_accur_list.append(test_accur//1)

            if i % print_freq == 0:
                # if i > (epochs//2):
                if i > (epochs//2) and top_train_accur < train_accur:
                    top_train_accur = train_accur
                if i > (epochs//2) and top_test_accur < test_accur:
                    top_test_accur = test_accur
                    print(top_test_accur)
                print('='*50)
                print('epoch                : ',i, '/',epochs)
                print('batch loss           : ',batch_loss)
                print("Training Accuracy (%): ", train_accur)
                print("Test Accuracy     (%): ", test_accur)
                print('pred                 :', np.transpose(np.argmax(preds_nparray, 1)))
                print('label                :', np.transpose(np.argmax(Y_test, 1)))

        writer.close()
        print('<< top accuracy >>')
        print('Training : ', top_train_accur)
        print('Testing  : ', top_test_accur)

        for i in range(len(train_accur_list)):
            print(train_accur_list[i] , test_accur_list[i])

''' 
string = 'epoch : '
print('{:>10}'.format(string))
print('%-10s' % ('test',))
assert False
'''
import numpy as np
import tensorflow as tf
data = np.array([[0,0],[1,0],[0,1],[1,1]])
label = np.array([[0],[1],[1],[0]])

# build model
feature_num = 10
output_dim = 4
stddev = 0.1

learning_rate = 1 #2
X = tf.placeholder(tf.float32, [None, feature_num])
Y_ = tf.placeholder(tf.float32, [None, 4])
layer = [1024, 2048, output_dim]

# batch normalization, change activation functions
W1 = tf.Variable(tf.truncated_normal([feature_num, layer[0]], mean=0, stddev=stddev), name="FCN1")
W2 = tf.Variable(tf.truncated_normal([layer[0], layer[1]], mean=0, stddev=stddev), name="FCN2")
W3 = tf.Variable(tf.truncated_normal([layer[1], layer[2]], mean=0, stddev=stddev), name="FCN3")
B1 = tf.Variable(tf.zeros([layer[0]]))
B2 = tf.Variable(tf.zeros([layer[1]]))
B3 = tf.Variable(tf.zeros([layer[2]]))

activation=tf.nn.sigmoid
prediction=tf.nn.softmax
# activation=tf.nn.relu
H1 = activation(tf.matmul(X,W1) + B1)
H2 = activation(tf.matmul(H1,W2) + B2)
H3 = prediction(tf.matmul(H2,W3) + B3)
Y = H3

cost = tf.reduce_sum(tf.squared_difference(Y_, Y))
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y,labels=Y))
# cost = - tf.reduce_mean(Y_*tf.log(Y)+(1-Y_)*tf.log(1.0-Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate)# why it learn in a wrong way if i append minimize(cost) directly ??
# optimizer = tf.train.AdagradOptimizer(learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train = optimizer.minimize(cost)

is_correct = tf.equal(Y, Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 10000
for epoch in range(epochs):
    # for i in range(4):
        batch_data = data
        batch_label = label
        sess.run([train], feed_dict={X: batch_data, Y_: batch_label})
        cost_val, output = sess.run([cost, Y], feed_dict={X: batch_data, Y_: batch_label})
        for i in range(4):
            print(batch_data[i], output[i], batch_label[i])
        print('Train result | Epoch:', '%04d' % (epoch + 1), '| 정확도:', sess.run(accuracy, feed_dict={X: batch_data, Y_: batch_label}),'Avg. cost =', '{:.3f}'.format(cost_val))

test_data = data
test_label = label
for i in range(len(test_data)):
    data, label = test_data[i], test_label[i]
    data = np.expand_dims(data ,0)
    output = sess.run(Y, feed_dict={X : data})
    print('The output of {} : {} | label : {}'.format(data, output, label))


'''
Basic Code for a convolutional neural network with 2 conv layers, a max pool layer, and 2 full-connected layers
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 128
display_step = 100

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10
NUM_C1 = 96 
NUM_C2 = 256
NUM_C3 = 384
NUM_C4 = 384
NUM_C5 = 256
NUM_H1 = 512
NUM_H2 = 256

# Put in the reponse normalization layers and figure out the proper numbers to optimize the algorithm, also put in the dropout in the fully connected layers

# Network Variables and placeholders
X = tf.placeholder(tf.float32, [None, NUM_INPUTS])  # Input
Y = tf.placeholder(tf.float32, [None, NUM_OUTPUTS]) # Truth Data - Output
isTraining = tf.placeholder(tf.bool)

# Network Architecture
def network():
    # Reshape to match picture format [BatchSize, Height x Width x Channel] => [Batch Size, Height, Width, Channel]
    x = tf.reshape(X, shape=[-1, 28, 28, 1])

    # Convolutional layers and max pool
    he_init = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.conv2d(x, NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    #reponse normalization here
   #tf.nn.local_response_normalization( )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, NUM_C2, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    #reponse normalization here
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, NUM_C3, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h3')
    conv4 = tf.layers.conv2d(conv3, NUM_C4, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h4')
    conv5 = tf.layers.conv2d(conv4, NUM_C5, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h5')
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(pool1)

    # Fully-connected layers 
    fc1 = tf.layers.dense(flatten, NUM_H1, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')   # First hidden layer with relu
    fc2 = tf.layers.dense(fc1, NUM_H2, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2') # Second hidden layer with relu
    fc3 = tf.layers.dense(fc1, NUM_H2, activation=tf.nn.relu, kernel_initializer=he_init, name='fc3') # Second hidden layer with relu

    print(x, conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool3, fc1, fc2, fc3)

    logits = tf.layers.dense(fc2, NUM_OUTPUTS, name='logits')  # this tf.layers.dense is same as tf.matmul(x, W) + b
    prediction = tf.nn.softmax(logits)
    return logits, prediction

# Define loss and optimizer
logits, prediction = network()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run( trainer, feed_dict={X: batch_xs, Y: batch_ys, isTraining:True })

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, isTraining:False})
      _step.append(step)
      _acc.append(acc)

      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for MINST Classification")
plt.show()








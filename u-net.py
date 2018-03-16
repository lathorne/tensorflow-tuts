import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os

#input dataset
from dataloader import DataLoader
dataset = DataLoader()

#Image input parameters, make these square
IM_WIDTH = 80 #must match image size in dataloader.py
IM_HEIGHT = 80 #must match image size in dataloader.py
IM_CHANNELS = 3

# Training Parameters
learning_rate = 0.0001
num_steps = 1000
batch_size = 32
display_step = 50

NUM_INPUTS = IM_WIDTH * IM_HEIGHT * IM_CHANNELS #downsize the images 64x64 and then change this number
NUM_OUTPUTS = 2 #number of output channels
NUM_C1 = 32 #size of conv layer, first layer should be smaller or the same size of your image
NUM_C2 = 32

# Network Variables and placeholders
X = tf.placeholder(tf.float32, [None, IM_WIDTH, IM_HEIGHT, IM_CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, IM_WIDTH, IM_HEIGHT, NUM_OUTPUTS]) # Truth Data - Output
isTraining = tf.placeholder(tf.bool)

def network():

 	#first two conv layers - add more layers here
  he_init = tf.contrib.layers.variance_scaling_initializer()
  conv1 = tf.layers.conv2d(X,      NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
  conv2 = tf.layers.conv2d(conv1,  NUM_C2, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
  conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
  up1 = tf.layers.conv2d_transpose(conv4, 64, [3, 3], strides=2, padding="SAME", name='Up1')

  conv5 = tf.layers.conv2d(up1,   NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
  conv6 = tf.layers.conv2d(conv5, NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
  logits = tf.layers.conv2d(conv6, NUM_OUTPUTS, [1, 1], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='out')
  prediction = tf.nn.softmax(logits)
  return logits, prediction

# Define loss and optimizer
logits, prediction = network()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) #need Y and logits to be the same size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model - change for segmentation - need better accuracy measure
correct_pred = tf.equal(tf.argmax(prediction, 3), tf.argmax(Y, 3)) #index three of prediction and Y gets us the fourth channel to compare
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
segmentation = tf.argmax(prediction, 3)

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = dataset.next_batch(batch_size) #grabs the training data and its truth value here
    sess.run( trainer, feed_dict={X: batch_xs, Y: batch_ys} )

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={X: dataset.x_test, Y:dataset.y_test}) #grabs the test data and its truth value, not the same as the training data
      #generate the test data at the beginning and leave it the same the whole: 60:40 or 80:20 training to test data
      _step.append(step)
      _acc.append(acc)

      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for Classification")
plt.show()

segmentation = 1 - sess.run(segmentation, feed_dict={X: dataset.x_test, Y:dataset.y_test})
index = 0;
matplotlib.image.imsave('real-img.png', dataset.x_test[index], cmap='gray') 
matplotlib.image.imsave('real-test.png', dataset.y_test[index][:,:,0], cmap='gray') 
matplotlib.image.imsave('real-results.png', segmentation[index], cmap='gray') 

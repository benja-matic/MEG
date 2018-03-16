#load data
from numpy import *
X=load('Xtr.npy')
y=load('ytr.py')
#TF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#define model
x = tf.placeholder(tf.float32, shape=[None, Nc])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

#model 1
# W = weight_variable([Nc, 2])
# b = bias_variable([2])

#model 2
W1 = weight_variable([Nc, 10*Nc])
b1 = bias_variable([10*Nc])
h1=tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable([10*Nc, 10*Nc])
b2 = bias_variable([10*Nc])
h2=tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = weight_variable([10*Nc, 2])
b3 = bias_variable([2])

y = tf.nn.softmax(tf.matmul(h2, W3) + b3) #prediction layer

# ##define traing, validation data and run regression
# Xtr=X[:ntr,:]
# labelstr=labels[:ntr]
#
# Xv=X[ntr:,:]
# labelsv=labels[ntr:]

# cross_entropy = tf.reduce_mean(
    # tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

loss = tf.reduce_mean(tf.squared_difference(y, y_))

l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=.9, scope=None
)

theta = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, theta)

regularized_loss = loss + regularization_penalty


train_step = tf.train.AdamOptimizer(1e-4).minimize(regularized_loss)#(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ntr=10
ind=range(len(Xtr))
random.shuffle(ind)
Xtr2=Xtr[ind[:ntr],:]
labelstr2=labelstr[ind[:ntr],:]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10001):
    if i % 100 == 0:
        random.shuffle(ind)
        Xtr2=Xtr[ind[:ntr],:]
        labelstr2=labelstr[ind[:ntr],:]
        print('step %d,test accuracy %g' % (i,accuracy.eval(feed_dict={
        x: Xv, y_: labelsv})))
      # train_accuracy = accuracy.eval(feed_dict={
      #     x: Xtr, y_: labelstr})
      # print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: Xtr2, y_: labelstr2})
    a=sess.run(W2)

  # print('test accuracy %g' % accuracy.eval(feed_dict={
  #     x: Xv, y_: labelsv}))

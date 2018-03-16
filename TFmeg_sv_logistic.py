#load data
from numpy import *
from scipy.stats import *
Nc=272
fs=1200
t0=0
t1=200
Xtr=mean((zscore(load('Xtr.npy')[:,:,t0:t1],1))**2,2)
ytr=load('ytr.npy')
Xv=mean((zscore(load('Xv.npy')[:50,:,:],1))**2,2)
yv=load('yv.npy')[:50,:]

ttot=1#int(1*1200)
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
x = tf.placeholder(tf.float32, shape=[None, Nc])#,ttot])
x_flat = tf.reshape(x, [-1, Nc*ttot])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# model 1
W = weight_variable([Nc*ttot, 2])
b = bias_variable([2])

y = tf.nn.softmax(tf.matmul(x_flat, W) + b) #prediction layer


# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
loss = tf.reduce_mean(tf.squared_difference(y, y_))
#
l2_regularizer = tf.contrib.layers.l2_regularizer(
   scale=1e-3, scope=None
)

theta = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, theta)
#
regularized_loss = loss + regularization_penalty


train_step = tf.train.AdamOptimizer(1e-4).minimize(regularized_loss)#(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10001):
    if i % 10 == 0:
        print('step %d,train accuracy %g' % (i,accuracy.eval(feed_dict={
        x: Xtr, y_: ytr})))
        print('step %d,test accuracy %g' % (i,accuracy.eval(feed_dict={
        x: Xv, y_: yv})))
      # train_accuracy = accuracy.eval(feed_dict={
      #     x: Xtr, y_: labelstr})
      # print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: Xtr, y_: ytr})
    # a=sess.run(W2)

  # print('test accuracy %g' % accuracy.eval(feed_dict={
  #     x: Xv, y_: labelsv}))

#dominhkha

import numpy as np 
import tensorflow as tf 
import os

cwp=os.getcwd()
X_train=np.array([i for i in range(0,20)])
X_train=X_train.reshape([20,1])
y_train=np.array(10*X_train+1)
y_train=y_train.reshape([20,1])

x=tf.placeholder(tf.float32,shape=[None,1])
y=tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(np.random.normal(size=[1,1]),dtype=tf.float32)
b = tf.Variable(np.random.normal(size=[1]),dtype=tf.float32)

y_hat = tf.add(tf.matmul(x,W),b)

cost=tf.reduce_sum(tf.square(y-y_hat))

train_step=tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
feed_dict={x:X_train,y:y_train}

for i in range(100):
    sess.run(train_step,feed_dict=feed_dict)
    print("Iteration {}, W: {}, b: {} ".format(i,sess.run(W),sess.run(b)))
    print("loss: {}".format(sess.run(cost,feed_dict=feed_dict)))


x_test=np.expand_dims(np.array([9]),axis=0)
print("predict: {}".format(sess.run(y_hat,feed_dict={x:x_test})))
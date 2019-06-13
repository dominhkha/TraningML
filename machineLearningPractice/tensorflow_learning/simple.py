# I learn from book: Hands on machine ....
import tensorflow as tf 
import numpy as np 
import pandas as pd 

# graph the variables
x=tf.Variable(3,name='x')
y=tf.Variable(4,name='y')
f=x*x*y+y*y

# solution 1
sess=tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
reult=sess.run(f)

# solution 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result=f.eval()
    print(result)

# solution 3

init=tf.global_variables_initializer()
with tf.Session as sess:
    init.run()
    result=f.eval()

# all above are default graph
# To make independent graph :
graph=tf.Graph()
with graph.as_default():
    x2=tf.Variable(2)
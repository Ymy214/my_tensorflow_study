#!/usr/bin/env python
# _*_ coding:utf-8 _*_


import tensorflow as tf
import numpy as np
import matplotlib

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.5+0.2

# create tensorflow structure start
# 参数变量
Weigths = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weigths*x_data+biases
# 计算差别
loss = tf.reduce_mean(tf.square(y-y_data))
# 0.5是学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# create tensorflow structure start

sess = tf.Session()
# 重要
sess.run(init)


for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(Weigths), sess.run(biases))



plt = tf.placeholder()








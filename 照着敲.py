#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a = tf.constant([1,2,3,4,5])
a_con = tf.constant(3.0)
b_con = tf.constant(4.0)

ab_sub = tf.subtract(a_con, b_con, name="ab_sub")
c = tf.add(a_con, b_con)


# 创建一个正太随机变量 2*3
var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0, name="var"))
print(a, var)
# 添加一个初始化所有变量的op，显式的初始化操作，将所有op初始化
init_op = tf.global_variables_initializer()

x = tf.constant([[-2, 3], [4, -5]], name="x_con")
x_abs = tf.abs(x, name="abs_op")

with tf.Session() as sess:
    # 并在绘画中开启（初始化所有变量的op）必须运行初始化
    # 也可以在回话中启动op并对tensor进行操作
    sess.run(init_op)
    sess.run(x_abs)
    sess.run(ab_sub)
    # sess.run(tf.abs(x))
    # 把程序的图结构写入事件文件，graph：吧指定的图写入文件事件中
    filewriter = tf.summary.FileWriter("./summary/graph/", graph=sess.graph)

    print(sess.run([a, var, c]))


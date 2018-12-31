#!/usr/bin/env python
# _*_ coding:utf-8 _*_


# 模拟异步子线程存入样本，主线程 读取

import tensorflow as tf
# 定义一个队列
Q = tf.FIFOQueue(3, tf.float32)
# 定义要做的事，循环+1 放入队列

# 放入一些数据
enq_many = Q.enqueue_many([[.1, .2, .3], ])

out_q = Q.dequeue()

data = out_q + 1

en_q = Q.enqueue(data)


with tf.Session as sess:
    sess.run(enq_many)

    for i in range(100):
        sess.run(en_q)


    for i in range(Q.size().eval()):
        print(sess.run(Q.dequeue()))











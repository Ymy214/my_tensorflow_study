#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def myregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    # 添加数据
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
    # 添加数据
    y_true = tf.matmul(x, [[0.7]]) + 0.8
    # 权重
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
    # 偏置
    bias = tf.Variable(0.0, name="b")

    y_predict = tf.matmul(x, weight) + bias

    # 建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 梯度下降优化损失率 learn_rate 学习率 一般指定[0,1]之间 ，取0.5
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化所有变量的op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 先初始化变量
        sess.run(init_op)

        # 打印出随机最先初始化的权重和偏置
        print("随机初始化的权重为：%f，偏置为：%f" % (weight.eval(), bias.eval()))

        # 循环进行 梯度下降 运行优化
        for i in range(101):

            sess.run(train_op)

            print("第%d次训练后更新的权重为：%f，偏置为：%f" % (i, weight.eval(), bias.eval()))

    # 将程序运行的结构图写入到文件当中去
    # filewriter = tf.summary.FileWriter("./summary/myregression/", graph=sess.graph)
    return None


if __name__ == "__main__":
    myregression()




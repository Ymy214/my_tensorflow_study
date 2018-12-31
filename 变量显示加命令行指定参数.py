#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
# import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 第一个参数：名字，默认值，说明
tf.app.flags.D

def myregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        # 添加数据
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
        # 添加数据
        y_true = tf.matmul(x, [[0.7]]) + 0.8
        # 权重
    with tf.variable_scope("model"):
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
        # 偏置
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 梯度下降优化损失率 learn_rate 学习率 一般指定[0,1]之间 ，取0.5
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 收集tenshor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)
    # 定义合并tensor的op
    merged = tf.summary.merge_all()
    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 定义一个初始化所有变量的op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 先初始化变量
        sess.run(init_op)

        # 打印出随机最先初始化的权重和偏置
        print("随机初始化的权重为：%f，偏置为：%f" % (weight.eval(), bias.eval()))
        filewriter = tf.summary.FileWriter("./summary/losses", graph=sess.graph)

        # 加载模型(先判断是否存在模型)，覆盖模型当中随即定义的参数，从上次训练的参数结果开始
        if os.path.exists("./checkpoint/checkpoint"):
            saver.restore(sess, "./checkpoint/model")

        # 循环进行 梯度下降 运行优化
        for i in range(300):

            sess.run(train_op)
            # 运行合并的tensor
            summary = sess.run(merged)
            filewriter.add_summary(summary, i)

            print("第%d次训练后更新的权重为：%f，偏置为：%f" % (i, weight.eval(), bias.eval()))
        # 保存模型，下次训练可以从保存的模型开始
        saver.save(sess, "./checkpoint/model")

    return None


if __name__ == "__main__":
    myregression()




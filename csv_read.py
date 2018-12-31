#!/usr/bin/env python
# _*_ coding:utf-8 _*_


import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def csvread(filelist):
    """
    读取csv文件
    :return: 读取的内容
    """
    # 1、构造文件列表器
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造csv文件列表器读取队列数据（按行读取）
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)

    # 3、对每行解码
    records = [["None"], [tf.int64]]
    # 想要读取多个数据。就用批处理

    example, label = tf.decode_csv(value, record_defaults=records)

    example_batch, label_batch = tf.train.batch([example, label], batch_size=9)

    print(example_batch, label_batch)
    return example_batch, label_batch






if __name__ == "__main__":
    file_name = os.listdir("./")


    csvread()




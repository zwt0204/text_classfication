# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/19 14:41
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import json
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class Model_Class:
    def __init__(self, vocab_file, gpu_num=2, is_training=True):
        self.is_training = is_training
        self.nclassnames = {"肯定": 1, "否定": 0}
        self.classnames = {1: "肯定", 0: "否定"}
        self.num_classes = len(self.classnames)
        self.sequence_length = 70
        self.num_filters = 128
        self.embedding_size = 256
        self.learning_rate = 0.0005
        self.clip = 5.0
        self.gpu_num = gpu_num
        self.MOVING_AVERAGE_DECAY = 0.99
        self.kernel_size = [2, 3, 4]
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        with tf.name_scope("classification_declare"):
            self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='input_y')
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.create_embedding()
        self.create_model()
        if self.is_training is True:
            self.create_loss()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def create_embedding(self):
        with tf.name_scope("classification_declare"):
            # 获取词向量
            self.embedding = tf.get_variable("embeddings", shape=[self.vocab_size, self.embedding_size],
                                             initializer=xavier_initializer())
            #  [None, sequence_length, embedding_size]
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # 增加一个维度 tf.nn.conv2d的输入为[batch, in_height, in_width, in_channels]
            self.embedding_expand = tf.expand_dims(embedding_input, -1)

    def create_model(self):
        with tf.name_scope("classification_cnn"):
            # 梯度保存
            self.tower_grads = []
            self.reuse_variables = False

            pooled_outputs = []
            # kernel_size滑动窗口的大小
            # num_filters卷积核个数
            # 128个2*100， 3*100， 4*100
            # 相当于以2， 3， 4gram来对句子进行了特征提取
            for i, filter_size in enumerate(self.kernel_size):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # [kernel_size, 100, 1, 128]
                    # kernel_size个厚度为100的1*128的三维张量
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    # filter： w，要求也是一个张量，shape为[filter_height, filter_weight, in_channel, out_channels]，
                    # 其中 filter_height为卷积核高度，
                    # filter_weight为卷积核宽度
                    # in_channel是图像通道数 ，和input的in_channel要保持一致
                    # out_channel是卷积核数量。
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                    # strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步
                    # padding采用的方式是VALID 输出高度 输入维度-kernel_size+1后除以步长
                    conv = tf.nn.conv2d(self.embedding_expand, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')
                    pooled_outputs.append(pooled)

            num_filter_total = self.num_filters * len(self.kernel_size)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_pro)

            with tf.name_scope('output'):
                w = tf.get_variable("w", shape=[num_filter_total, self.num_classes],
                                    initializer=xavier_initializer())

                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')

                self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
                self.pro = tf.nn.softmax(self.scores)
                self.predicitions = tf.argmax(self.pro, 1, name='predictions')

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def create_loss(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        with tf.name_scope("classification_loss"):
            # 将神经网络的优化过程跑在不同的GPU上。
            for i in range(self.gpu_num):
                with tf.device('/gpu:%d' % i):
                    # 将优化过程指定在一个GPU上。
                    with tf.name_scope('GPU_%d' % i) as scope:
                        self.cost_func = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))
                        self.reuse_variables = True
                        grads = self.optimizer.compute_gradients(self.cost_func)
                        self.tower_grads.append(grads)
            # 计算变量的平均梯度。
            self.grads = self.average_gradients(self.tower_grads)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # 使用平均梯度更新参数。
            self.apply_gradient_op = optimizer.apply_gradients(self.grads, global_step=self.global_step)

            # 计算变量的滑动平均值。
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, self.global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)
            # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
            self.train_op = tf.group(self.apply_gradient_op, variables_averages_op)

            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_func)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  # 对交叉熵取均值非常有必要

        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predicitions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')

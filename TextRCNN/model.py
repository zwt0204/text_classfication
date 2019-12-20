# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/20 9:39
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import json
import tensorflow as tf


class Model_Class:
    def __init__(self, vocab_file, keep_prob=0.5, is_training=True):
        self.is_training = is_training
        self.nclassnames = {"肯定": 1, "否定": 0}
        self.classnames = {1: "肯定", 0: "否定"}
        self.num_classes = len(self.classnames)
        self.sequence_length = 70
        self.hidden_size = 64
        self.embedding_size = 200
        self.keep_prob = keep_prob
        self.learning_rate = 0.0005
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.text_hidden_size = 64
        self.num_layers = 2
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        with tf.name_scope("classification_declare"):
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
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
            self.word = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name='W'
            )

    def cells(self, size, reuse=False):
        # reuse:布尔类型，描述是否在现有范围中重用变量。如果不为True，并且现有范围已经具有给定变量，则会引发错误。
        # tf.orthogonal_initializer函数：正交矩阵的初始化器(要初始化的张量的形状是二维)
        return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def create_model(self):
        # shape:[None,sentence_length,embed_size]
        with tf.name_scope("classification_lstm"):
            self.embedded_chars = tf.nn.embedding_lookup(self.word, self.input_x)

            for n in range(self.num_layers):
                (self.output_fw, self.output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cells(self.hidden_size // 2),
                    cell_bw=self.cells(self.hidden_size // 2),
                    inputs=self.embedded_chars,
                    dtype=tf.float32,
                    scope='bidirectional_rnn_%d' % (n))

        with tf.name_scope('context'):
            # 获取左右两边上下文嵌入
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name='context_left')
            self.c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name='context_right')

        # word representation
        with tf.name_scope('word-representation'):
            # 组合生成最后的词向量
            self.x = tf.concat([self.c_left, self.embedded_chars, self.c_right], axis=2, name='x')
            embedding_size = 2 * self.hidden_size + self.embedding_size

        # text representation
        with tf.name_scope('text_representation'):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.text_hidden_size], -1.0, 1.0), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[self.text_hidden_size]), name='b2')
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        # max pooling
        with tf.name_scope('max_pooling'):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        # final scores and predictions
        with tf.name_scope('output'):
            W4 = tf.get_variable(
                "W4",
                shape=[self.text_hidden_size, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b4')
            self.scores = tf.nn.xw_plus_b(self.y3, W4, b4, name='scores')
            self.predictions = tf.nn.softmax(self.scores, name='predictions')

    def create_loss(self):
        # loss
        with tf.name_scope('loss'):
            self.cost_func = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.predictions), tf.round(self.input_y)), tf.float32), name="Accuracy")
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_func)

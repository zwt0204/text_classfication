# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/20 13:05
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import json
import tensorflow as tf
from XL import transformer


class Model_Classs:
    def __init__(self, vocab_file, keep_prob=0.5, is_training=True):
        self.is_training = is_training
        self.nclassnames = {"肯定": 1, "否定": 0}
        self.classnames = {1: "肯定", 0: "否定"}
        self.num_classes = len(self.classnames)
        self.sequence_length = 70
        self.embedding_size = 300
        self.keep_prob = keep_prob
        self.learning_rate = 0.0001
        self.n_layer = 1
        self.d_model = 256
        self.n_head = 30
        self.d_head = 50
        self.d_inner = 512
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.PAD = self.char_index['PAD']
        self.vocab_size = len(self.char_index) + 1
        with tf.name_scope("classification_declare"):
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.create_model()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def create_model(self):
        with tf.name_scope("classification_lstm"):
            self.memory = tf.fill([self.n_layer,
                                   tf.shape(self.input_x)[0],
                                   tf.shape(self.input_x)[1],
                                   self.d_model], self.PAD)
            self.memory = tf.cast(self.memory, tf.float32)
            initializer = tf.initializers.random_normal(stddev=2 / self.vocab_size)
            logits, self.next_memory = transformer(
                self.input_x,
                self.memory,
                self.vocab_size,
                self.n_layer,
                self.d_model,
                self.embedding_size,
                self.n_head,
                self.d_head,
                self.d_inner,
                initializer
            )
            logits = tf.reduce_mean(logits, axis=1)
            self.logits = tf.layers.dense(logits, units=self.num_classes)
            self.prediction = tf.nn.softmax(self.logits)

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.prediction), tf.round(self.input_y)), tf.float32), name="Accuracy")

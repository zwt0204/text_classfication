#!/user/bin/env python
# coding=utf-8
"""
@file: test.py
@author: zwt
@time: 2020/11/24 15:45
@desc: 
"""
import numpy as np
from model6 import Model_Class
import tensorflow as tf
import logging
import json


class ModelPredicter:

    def __init__(self):
        self.model_dir = 'models/text'
        self.graph = tf.Graph()
        self.keep_prob = 1.0
        self.is_training = False
        self.num_class = 86
        self.vocab_file = 'data/vocab.txt'
        with self.graph.as_default():
            with tf.variable_scope('classification_query'):
                self.model = Model_Class(self.vocab_file, num_class=self.num_class, is_training=self.is_training)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            logging.info("load classification success...")
        else:
            raise Exception("load classification failure...")

    def predict(self, input_text):
        """
        预测文本
        :param input_text:
        :return:
        """
        char_vector = self.convert_vector(input_text, self.model.sequence_length)
        feed_dict = {self.model.input_x: char_vector, self.model.keep_pro: self.keep_prob}
        values = self.session.run(self.model.predictions, feed_dict)
        idx = np.argwhere(values[0] == 1)
        return idx
        # values = self.session.run(self.model.probabilities, feed_dict)
        # return values

    def convert_vector(self, input_text, limit):
        char_vector = np.zeros((self.model.sequence_length), dtype=np.float32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]

        return np.array([char_vector])


if __name__ == '__main__':
    predict = ModelPredicter()
    res = predict.predict('data')
# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/12/19 10:55
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from datetime import datetime
import numpy as np
from model import Moldel_Class
import tensorflow as tf
import logging
import json
from sklearn.metrics import precision_score


class ModelPredicter:
    def __init__(self):
        self.model_dir = '../model/text'
        self.graph = tf.Graph()
        self.keep_prob = 1.0
        self.is_training = False
        self.vocab_file = '../data/dictionary.json'
        with self.graph.as_default():
            with tf.variable_scope('classification_query'):
                self.model = Moldel_Class(self.vocab_file, keep_prob=self.keep_prob, is_training=self.is_training)
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

        x_test = []
        y_test = []
        with open('../data/dev.json', 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                temp = self.convert_vector(data['sentence'].strip(), self.model.sequence_length)
                y_test.append(int(data['label']))
                x_test.append(temp)

        x_test = np.array(x_test).reshape((10000, 70))
        feed_dict = {self.model.char_inputs: x_test}
        values = self.session.run(self.model.prediction, feed_dict)

        precision_score(y_test, [np.argmax(i) for i in values], average='micro')

        char_vector = self.convert_vector(input_text, self.model.sequence_length)
        t1 = datetime.now()
        feed_dict = {self.model.char_inputs: char_vector}
        values = self.session.run(self.model.prediction, feed_dict)
        t2 = datetime.now()
        logging.debug("class predict time : %f" % ((t2 - t1).microseconds / 1000.))
        idx = np.argmax(values[0])
        classname = self.model.classnames[idx]
        print(
            "input: {0} , predict classname {1} , classid {2} , probability {3}".format(input_text, classname, idx,
                                                                                        values[0][idx]))
        return idx, classname, values[0][idx], max(values[0])

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
    model_predict = ModelPredicter()
    model_predict.predict('江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物')
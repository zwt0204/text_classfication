# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/12/19 14:52
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from datetime import datetime
import numpy
from model import Model_Class
import tensorflow as tf
import logging


class ModelPredicter:
    def __init__(self):
        self.model_dir = 'text'
        self.graph = tf.Graph()
        self.keep_prob = 1.0
        self.is_training = False
        self.vocab_file = 'vocab.json'
        with self.graph.as_default():
            with tf.variable_scope('classification_query'):
                self.model = Model_Class(self.vocab_file, is_training=self.is_training)
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
        t1 = datetime.now()
        feed_dict = {self.model.input_x: char_vector, self.model.keep_pro: self.keep_prob}
        values = self.session.run(self.model.pro, feed_dict)
        t2 = datetime.now()
        logging.debug("class predict time : %f" % ((t2 - t1).microseconds / 1000.))
        idx = numpy.argmax(values[0])
        classname = self.model.classnames[idx]
        logging.info(
            "input: {0} , predict classname {1} , classid {2} , probability {3}".format(input_text, classname, idx,
                                                                                        values[0][idx]))
        return idx, classname, values[0][idx], max(values[0])

    def convert_vector(self, input_text, limit):
        char_vector = numpy.zeros((self.model.sequence_length), dtype=numpy.float32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]

        return numpy.array([char_vector])


if __name__ == '__main__':
    predict = ModelPredicter()
    predict.predict('data')
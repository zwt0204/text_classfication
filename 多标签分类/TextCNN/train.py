#!/user/bin/env python
# coding=utf-8
"""
@file: train.py
@author: zwt
@time: 2020/11/24 15:44
@desc: 
"""
import os, json, numpy
import numpy as np
from model import Model_Class
import tensorflow as tf
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class ModelTrainner:
    def __init__(self, is_training=True):
        self.class_graph = tf.Graph()
        self.model_dir = "models/text"
        self.batch_size = 32
        self.num_class = 86
        self.is_training = is_training
        self.gpu_number = 2
        if self.is_training is True:
            self.keep_prob = 0.3
        else:
            self.keep_prob = 1.0
        with tf.variable_scope('classification_query'):
            self.model = Model_Class(vocab_file="data/vocab.txt", num_class=self.num_class, gpu_num=self.gpu_number,
                                                is_training=self.is_training)
        self.saver = tf.train.Saver()

    def train(self, epochs=15):
        xitems, yitems = self.load_samples_test("new_data/train.txt")
        xitems_train = xitems[:math.floor(len(xitems)*0.8)]
        yitems_train = yitems[:math.floor(len(xitems)*0.8)]
        batch_count = int(len(xitems_train) / (self.batch_size * self.gpu_number))
        print("prepare data success ...")
        initer = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(initer)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt is not None and ckpt.model_checkpoint_path:
                self.saver.restore(session, ckpt.model_checkpoint_path)
            for epoch in range(epochs):
                train_loss_value = 0.
                train_accuracy_value = 0.
                for i in range(batch_count):
                    batch_xitems = xitems_train[
                                   i * self.batch_size * self.gpu_number:(i + 1) * self.batch_size * self.gpu_number]
                    batch_yitems = yitems_train[
                                   i * self.batch_size * self.gpu_number:(i + 1) * self.batch_size * self.gpu_number]
                    batch_char_inputs, batch_ys = self.convert_batch(batch_xitems, batch_yitems)
                    feed_dict = {self.model.input_x: batch_char_inputs, self.model.input_y: batch_ys,
                                 self.model.keep_pro: 0.5}
                    acc, batch_loss_value, _ = session.run([self.model.accuracy, self.model.cost_func, self.model.train_op], feed_dict)
                    train_loss_value += batch_loss_value / batch_count
                    train_accuracy_value += acc / batch_count
                    batch_buffer = "Progress {0}/{1} , cost : {2}, acc：{3}".format(i + 1, batch_count, batch_loss_value, train_accuracy_value)
                    if i % 200 == 0:
                        print(batch_buffer)
                print("Epoch: %d/%d , train cost=%f " % ((epoch + 1), epochs, train_loss_value))
                self.saver.save(session, os.path.join(self.model_dir, "textclassification.dat"))
            coord.request_stop()
            coord.join(threads)

    def read_sample_file(self, datafile):
        row_mapper = {}
        with open(datafile, "r+", encoding="utf-8") as reader:
            for line in reader:
                record = json.loads(line.strip().lower())
                classids = []
                for i in record['label']:
                    classid = int(i)
                    classids.append(classid)
                classids = json.dumps(classids)
                raw_text = record['query']

                row_mapper[raw_text.strip().lower()] = classids
        return row_mapper

    def load_samples_test(self, datafiles):
        import random
        xrows = []
        yrows = []
        text_mapper = self.read_sample_file(datafiles)

        for v, k in text_mapper.items():
            yvector = np.zeros((self.num_class), dtype=np.float32)
            for i in json.loads(k):
                yvector[i] = 1.0
            xrows.append(v)
            yrows.append(yvector)

        ncount = len(xrows)
        idx = np.random.choice(ncount, ncount, replace=False)
        xitems = np.array(xrows)[idx]
        yitems = np.array(yrows)[idx]
        xitems = list(xitems)
        yitems = list(yitems)

        temp = list(zip(xitems, yitems))
        random.shuffle(temp)
        xitems, yitems = zip(*temp)
        return xitems, yitems

    def convert_batch(self, xitems, yitems):
        xrecords = numpy.zeros((self.batch_size * 2, self.model.sequence_length))
        for i in range(len(xitems)):
            count = len(xitems[i])
            if count > self.model.sequence_length:
                count = self.model.sequence_length
            for j in range(count):
                if xitems[i][j] in self.model.char_index.keys():
                    xrecords[i][j] = self.model.char_index[xitems[i][j]]
        return xrecords, numpy.array(yitems, dtype=numpy.float32)

    def convert_vector(self, input_text, limit):
        char_vector = numpy.zeros((self.model.sequence_length), dtype=numpy.float32)
        count = len(input_text.strip().low())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]

        return numpy.array([char_vector])


if __name__ == "__main__":
    trainner = ModelTrainner(is_training=True)
    trainner.train(epochs=20)
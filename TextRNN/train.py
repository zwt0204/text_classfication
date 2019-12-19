# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/19 10:50
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os, json, numpy
from model import Moldel_Class
import tensorflow as tf

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ModelTrainner:
    def __init__(self, is_training=True):
        self.class_graph = tf.Graph()
        self.model_dir = "text"
        self.batch_size = 32
        self.is_training = is_training
        if self.is_training == True:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0
        with tf.variable_scope('classification_query'):
            self.model = Moldel_Class(vocab_file="vocab.json",
                                      keep_prob=self.keep_prob, is_training=self.is_training)
        self.saver = tf.train.Saver()

    def train(self, epochs=20):
        xitems, yitems = self.load_samples_test(datafiles='test.txt')
        batch_count = int(len(xitems) / self.batch_size)
        initer = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(initer)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt is not None and ckpt.model_checkpoint_path:
                self.saver.restore(session, ckpt.model_checkpoint_path)
            for epoch in range(epochs):
                train_loss_value = 0.
                train_accuracy_value = 0.
                for i in range(batch_count):
                    batch_xitems = xitems[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_yitems = yitems[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_char_inputs, batch_ys = self.convert_batch(batch_xitems, batch_yitems)
                    feed_dict = {self.model.char_inputs: batch_char_inputs, self.model.outputs: batch_ys}
                    batch_loss_value, acc, _ = session.run([self.model.cost_func, self.model.accuracy, self.model.optimizer], feed_dict)
                    train_loss_value += batch_loss_value / batch_count
                    train_accuracy_value += acc / batch_count

                    batch_buffer = "Progress {0}/{1} , cost : {2}".format(i + 1, batch_count, batch_loss_value)
                    if i % 20 == 0:
                        print(batch_buffer)
                print("Epoch: %d/%d , train cost=%f " % ((epoch + 1), epochs, train_loss_value))
                self.saver.save(session, os.path.join(self.model_dir, "textclassification.dat"))

    def read_sample_file(self, datafile):
        row_mapper = {}
        with open(datafile, "r+", encoding="utf-8") as reader:
            for line in reader:
                record = json.loads(line.strip().lower())
                classid = record['id']
                raw_text = record['text']
                if classid in row_mapper.keys():
                    row_mapper[classid].append(raw_text.strip().lower())
                else:
                    row_mapper[classid] = [raw_text.strip().lower()]
        return row_mapper

    def convert_batch(self, xitems, yitems):
        xrecords = numpy.zeros((self.batch_size, self.model.sequence_length))
        for i in range(len(xitems)):
            count = len(xitems[i])
            if count > self.model.sequence_length:
                count = self.model.sequence_length
            for j in range(count):
                if xitems[i][j] in self.model.char_index.keys():
                    xrecords[i][j] = self.model.char_index[xitems[i][j]]
        return xrecords, numpy.array(yitems, dtype=numpy.float32)

    def predict(self, input_text):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            saver = tf.train.Saver()
            if ckpt is not None and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print("load success...")
            char_vector = self.convert_vector(input_text, self.model.sequence_length)
            feed_dict = {self.model.char_inputs: char_vector}
            values = session.run(self.model.prediction, feed_dict)
            idx = numpy.argmax(values[0])
            classname = self.model.classnames[idx]
            return classname, values[0][idx]

    def predict_batch(self, session, input_text):
        char_vector = self.convert_vector(input_text, self.model.sequence_length)
        feed_dict = {self.model.char_inputs: char_vector}
        values = session.run(self.model.prediction, feed_dict)
        idx = numpy.argmax(values[0])
        classname = self.model.classnames[idx]
        return classname, values[0][idx]

    def convert_vector(self, input_text, limit):
        char_vector = numpy.zeros((self.model.sequence_length), dtype=numpy.float32)
        count = len(input_text.strip().lower())
        if count > limit:
            count = limit
        for i in range(count):
            if input_text[i] in self.model.char_index.keys():
                char_vector[i] = self.model.char_index[input_text[i]]

        return numpy.array([char_vector])

    def load_samples_test(self, datafiles):
        import random
        xrows = []
        yrows = []
        text_mapper = self.read_sample_file(datafiles)
        classnames = list(text_mapper.keys())
        classnames = list(set(classnames))
        for classid in classnames:
            row_items = []
            if classid in text_mapper.keys():
                row_items.extend(text_mapper[classid])
            # 相当于生成one-hot向量
            yvector = numpy.zeros((len(self.model.classnames)), dtype=numpy.float32)
            yvector[classid] = 1.0
            for random_item in row_items:
                yrows.append(yvector)
                xrows.append(random_item)
        ncount = len(xrows)
        idx = numpy.random.choice(ncount, ncount, replace=False)
        xitems = numpy.array(xrows)[idx]
        yitems = numpy.array(yrows)[idx]
        xitems = list(xitems)
        yitems = list(yitems)

        temp = list(zip(xitems, yitems))
        random.shuffle(temp)
        xitems, yitems = zip(*temp)
        return xitems, yitems


if __name__ == "__main__":
    trainner = ModelTrainner(is_training=True)
    trainner.train()

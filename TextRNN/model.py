# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/19 10:46
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import json
import tensorflow as tf
from attention import attention
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import xavier_initializer


class Moldel_Class:
    def __init__(self, vocab_file, keep_prob=0.5, is_training=True):
        self.is_training = is_training
        self.nclassnames = {"肯定": 1, "否定": 0}
        self.classnames = {1: "肯定", 0: "否定"}
        self.num_classes = len(self.classnames)
        self.sequence_length = 70
        self.hidden_size = 256
        self.embedding_size = 256
        self.keep_prob = keep_prob
        self.learning_rate = 0.0005
        self.layer_size = 1
        self.attention_size = 100
        self.warmup_steps = 1000
        self.clip = 5
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.vocab_size = len(self.char_index) + 1
        with tf.name_scope("classification_declare"):
            self.char_inputs = tf.placeholder(tf.int32, [None, self.sequence_length],
                                              name="char_inputs")
            self.outputs = tf.placeholder(tf.float32, [None, self.num_classes], name="outputs")
        self.create_embedding()
        self.create_model()
        if self.is_training == True:
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
            self.classification_variable = tf.get_variable("classification_embedding_variable",
                                                           shape=[self.vocab_size, self.embedding_size],
                                                           initializer=xavier_initializer())
            self.weight_variable = tf.get_variable("classification_weight_variable",
                                                   shape=[self.hidden_size * 2, self.num_classes],
                                                   initializer=xavier_initializer())
            self.bias_variable = tf.get_variable("classification_bias_variable", shape=[self.num_classes])

    def create_model(self):
        self.embedded_layer = tf.nn.embedding_lookup(self.classification_variable, self.char_inputs)
        with tf.name_scope("classification_rnn"):
            outputs, _ = bi_rnn(
                tf.nn.rnn_cell.DropoutWrapper(GRUCell(self.hidden_size), self.keep_prob),
                tf.nn.rnn_cell.DropoutWrapper(GRUCell(self.hidden_size), self.keep_prob),
                inputs=self.embedded_layer,
                dtype=tf.float32)

        # attention
        with tf.name_scope('Attention_layer'):
            attention_output, _, _ = attention(outputs, self.attention_size, return_alphas=True)

        self.drop = tf.nn.dropout(attention_output, self.keep_prob)
        #############################

        # 没有attention
        # outputs = tf.concat(outputs, axis=-1)
        # outputs = tf.reduce_mean(outputs, axis=1)
        # self.drop = tf.nn.dropout(outputs, self.keep_prob)
        ##############################

        self.logits = tf.matmul(self.drop, self.weight_variable) + self.bias_variable  # [batch_size,num_classes]
        self.prediction = tf.nn.softmax(self.logits)

    def create_loss(self):
        with tf.name_scope("classification_loss"):
            # 类别独立但是不互相排斥
            # self.cost_func = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.outputs, logits=self.logits))
            # 类别独立且排斥，一句话只能属于一个类别

            # self.outputs = self.label_smoothing(self.outputs)
            self.cost_func = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs, logits=self.logits))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.prediction), tf.round(self.outputs)), tf.float32), name="Accuracy")

            global_step = tf.train.get_or_create_global_step()
            lr = self.noam_scheme(self.learning_rate, global_step, self.warmup_steps)

            params = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_func, params), self.clip)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98,
                                                    epsilon=1e-8).apply_gradients(zip(grads, params))

    def noam_scheme(self, learning_rate, global_step, warm_steps=4000):
        """
        学习率预热：在训练的轮数达到warmup_steps过程中，学习率会逐渐增加到learning_rate，
        训练轮数超过warmup_steps之后学习率会从learning_rate开始逐步下降。
        :param learning_rate:
        :param global_step:
        :param warm_steps:
        :return:
        """
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return learning_rate * warm_steps ** 0.5 * tf.minimum(step * warm_steps ** -1.5, step ** -0.5)

    def label_smoothing(self, inputs, epsilon=0.1):
        """
        import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0], [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

        [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        :param inputs:
        :param epsilon:
        :return:
        """
        K = inputs.get_shape().as_list()[-1]
        return ((1 - epsilon) * inputs) + (epsilon / K)
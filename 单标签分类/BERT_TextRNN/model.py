# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/26 14:04
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import xavier_initializer
from BERT_TextCNN.al_bert import modeling


class Moldel_Class:
    def __init__(self, keep_prob=0.5, learning_rate=0.0005, is_training=True):
        self.is_training = is_training
        self.nclassnames = {"肯定": 1, "否定": 0}
        self.classnames = {1: "肯定", 0: "否定"}
        self.num_classes = len(self.classnames)
        self.sequence_length = 70
        self.hidden_size = 256
        self.embedding_size = 256
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.layer_size = 1
        self.attention_size = 100
        self.warmup_steps = 1000
        self.clip = 5
        self.vocab_file = './bert_model/chinese_L-12_H-768_A-12/vocab.txt'  # the path of vocab file
        self.bert_config_file = './bert_model/chinese_L-12_H-768_A-12/albert_config.json'  # the path of bert_cofig file
        self.init_checkpoint = './bert_model/chinese_L-12_H-768_A-12/albert_model.ckpt'  # the path of bert model
        self.use_one_hot_embeddings = False
        with tf.name_scope("classification_declare"):
            self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
            self.input_ids = tf.placeholder(tf.int64, shape=[None, self.sequence_length], name='input_ids')
            self.input_mask = tf.placeholder(tf.int64, shape=[None, self.sequence_length], name='input_mask')
            self.segment_ids = tf.placeholder(tf.int64, shape=[None, self.sequence_length], name='segment_ids')
            self.labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        self.create_embedding()
        self.create_model()
        if self.is_training is True:
            self.create_loss()

    def create_embedding(self):
        with tf.name_scope("classification_declare"):
            self.weight_variable = tf.get_variable("classification_weight_variable",
                                                   shape=[self.hidden_size * 2, self.num_classes],
                                                   initializer=xavier_initializer())
            self.bias_variable = tf.get_variable("classification_bias_variable", shape=[self.num_classes])

        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.use_one_hot_embeddings)
            self.embedded_layer = bert_model.get_pooled_output()

    def create_model(self):
        with tf.name_scope("classification_rnn"):
            outputs, _ = bi_rnn(
                tf.nn.rnn_cell.DropoutWrapper(GRUCell(self.hidden_size), self.keep_prob),
                tf.nn.rnn_cell.DropoutWrapper(GRUCell(self.hidden_size), self.keep_prob),
                inputs=self.embedded_layer,
                dtype=tf.float32)

        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.reduce_mean(outputs, axis=1)
        self.drop = tf.nn.dropout(outputs, self.keep_prob)

        self.logits = tf.matmul(self.drop, self.weight_variable) + self.bias_variable  # [batch_size,num_classes]
        self.prediction = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

    def create_loss(self):
        """计算loss，因为输入的样本标签不是one_hot的形式，需要转换下,可以加入标签平滑操作"""
        with tf.name_scope("classification_loss"):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.num_classes, dtype=tf.float32)
            one_hot_labels = self.label_smoothing(one_hot_labels)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.cost_func = tf.reduce_mean(per_example_loss)
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            self.global_step = tf.train.get_or_create_global_step()
            lr = self.noam_scheme(self.learning_rate, self.global_step, self.warmup_steps)

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
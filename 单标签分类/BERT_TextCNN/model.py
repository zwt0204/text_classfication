# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/20 13:40
@Author  : zwt
@git   : https://github.com/cjymz886/text_bert_cnn/blob/master/text_model.py
@Software: PyCharm
"""
import tensorflow as tf
from BERT_TextCNN.al_bert import modeling


class TextConfig:
    seq_length = 128  # max length of sentence
    num_labels = 10  # number of labels

    num_filters = 128  # number of convolution kernel
    filter_sizes = [2, 3, 4]  # size of convolution kernel
    hidden_dim = 128  # number of fully_connected layer units

    keep_prob = 0.5  # droppout
    lr = 5e-5  # learning rate
    lr_decay = 0.9  # learning rate decay
    clip = 5.0  # gradient clipping threshold

    is_training = True  # is _training
    use_one_hot_embeddings = False  # use_one_hot_embeddings

    num_epochs = 5  # epochs
    batch_size = 32  # batch_size
    print_per_batch = 200  # print result
    require_improvement = 1000  # stop training if no inporement over 1000 global_step

    output_dir = './result'
    data_dir = './corpus/cnews'  # the path of input_data file
    vocab_file = './bert_model/chinese_L-12_H-768_A-12/vocab.txt'  # the path of vocab file
    bert_config_file = './bert_model/chinese_L-12_H-768_A-12/albert_config.json'  # the path of bert_cofig file
    init_checkpoint = './bert_model/chinese_L-12_H-768_A-12/albert_model.ckpt'  # the path of bert model


class TextCNN(object):

    def __init__(self, config):
        '''获取超参数以及模型需要的传入的5个变量，input_ids，input_mask，segment_ids，labels，keep_prob'''
        self.config = config
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        self.input_ids = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int64, shape=[None, self.config.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()

    def cnn(self):
        '''获取bert模型最后的token-level形式的输出(get_sequence_output)，将此作为embedding_inputs，作为卷积的输入'''
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.config.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)
            embedding_inputs = bert_model.get_sequence_output()

        """conv1d： 用三个不同的卷积核进行卷积和池化，最后将三个结果concat"""
        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                    conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size, name='conv1d')
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 1)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])

        # conv2d： 用三个不同的卷积核进行卷积和池化，最后将三个结果concat
        # with tf.name_scope("conv"):
        #     pooled_outputs = []
        #     for i, filter_size in enumerate(self.config.filter_sizes):
        #         with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
        #             embedding_inputs_conv = tf.expand_dims(embedding_inputs, -1)
        #             filter_shape = [filter_size, embedding_inputs[-1], 1, self.config.num_filters]
        #             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #             b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
        #             conv = tf.nn.conv2d(
        #                 embedding_inputs_conv,
        #                 W,
        #                 strides=[1, 1, 1, 1],
        #                 padding="VALID",
        #                 name="conv")
        #             h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #             pooled = tf.nn.max_pool(
        #                 h,
        #                 ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
        #                 strides=[1, 1, 1, 1],
        #                 padding='VALID',
        #                 name="pool")
        #             pooled_outputs.append(pooled)
        #     num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        #     h_pool = tf.concat(pooled_outputs, 1)
        #     outputs = tf.reshape(h_pool, [-1, num_filters_total])

        """加全连接层和dropuout层"""
        with tf.name_scope('fc'):
            fc = tf.layers.dense(outputs, self.config.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_labels, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        """计算loss，因为输入的样本标签不是one_hot的形式，需要转换下,可以加入标签平滑操作"""
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

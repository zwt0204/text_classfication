# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/26 14:37
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import sys
import time
from sklearn import metrics
from model import Moldel_Class
from BERT_TextCNN.data_process import *


class ModerTrain:

    def __init__(self):
        self.output_dir = 'output'
        self.data_dir = 'data'
        self.batch_size = 128
        self.print_per_batch = 200
        self.require_improvement = 1000
        self.lr_decay = 0.9
        self.label_list = TextProcessor().get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.model.vocab_file, do_lower_case=False)
        self.model = Moldel_Class()

    def optimistic_restore(self, session, save_file):
        """载入bert模型"""
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for
                            var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                else:
                    print("variable not trained.var_name:", var_name)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def feed_data(self, batch_ids, batch_mask, batch_segment, batch_label, keep_prob):
        """构建text_model需要传入的数据"""
        feed_dict = {
            self.model.input_ids: np.array(batch_ids),
            self.model.input_mask: np.array(batch_mask),
            self.model.segment_ids: np.array(batch_segment),
            self.model.labels: np.array(batch_label),
            self.model.keep_prob: keep_prob
        }
        return feed_dict

    def train(self, epochs=10):
        """训练模型text_bert_cnn模型"""

        tensorboard_dir = os.path.join(self.output_dir, "tensorboard/textcnn")
        save_dir = os.path.join(self.output_dir, "checkpoints/textcnn")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'best_validation')

        start_time = time.time()

        tf.logging.info("*****************Loading training data*****************")
        train_examples = TextProcessor().get_train_examples(self.data_dir)
        trian_data = convert_examples_to_features(train_examples, self.label_list, self.model.sequence_length, self.tokenizer)

        tf.logging.info("*****************Loading dev data*****************")
        dev_examples = TextProcessor().get_dev_examples(self.data_dir)
        dev_data = convert_examples_to_features(dev_examples, self.label_list, self.model.sequence_length, self.tokenizer)

        tf.logging.info("Time cost: %.3f seconds...\n" % (time.time() - start_time))

        tf.logging.info("Building session and restore bert_model...\n")
        session = tf.Session()
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())

        tf.summary.scalar("loss", self.model.cost_func)
        tf.summary.scalar("accuracy", self.model.accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(session.graph)
        self.optimistic_restore(session, self.model.init_checkpoint)

        tf.logging.info('Training and evaluating...\n')
        best_acc = 0
        last_improved = 0  # record global_step at best_val_accuracy
        flag = False

        for epoch in range(epochs):
            batch_train = batch_iter(trian_data, self.batch_size)
            start = time.time()
            tf.logging.info('Epoch:%d' % (epoch + 1))
            for batch_ids, batch_mask, batch_segment, batch_label in batch_train:
                feed_dict = self.feed_data(batch_ids, batch_mask, batch_segment, batch_label, self.model.keep_prob)
                global_step, train_summaries, train_loss, train_accuracy = session.run(
                    [self.model.global_step,
                     merged_summary, self.model.cost_func,
                     self.model.accuracy], feed_dict=feed_dict)
                if global_step % self.print_per_batch == 0:
                    end = time.time()
                    val_loss, val_accuracy = self.evaluate(session, dev_data)
                    merged_acc = (train_accuracy + val_accuracy) / 2
                    if merged_acc > best_acc:
                        saver.save(session, save_path)
                        best_acc = merged_acc
                        last_improved = global_step
                    tf.logging.info(
                        "step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, "
                        "val accuracy: {:.3f},training speed: {:.3f}sec".format(
                            global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                            (end - start) / self.print_per_batch))
                    start = time.time()

                if global_step - last_improved > self.require_improvement:
                    tf.logging.info("No optimization over 1500 steps, stop training")
                    flag = True
                    break
            if flag:
                break
            self.model.learning_rate *= self.lr_decay

    def evaluate(self, sess, dev_data):
        """批量的形式计算验证集或测试集上数据的平均loss，平均accuracy"""
        data_len = 0
        total_loss = 0.0
        total_acc = 0.0
        for batch_ids, batch_mask, batch_segment, batch_label in batch_iter(dev_data, self.batch_size):
            batch_len = len(batch_ids)
            data_len += batch_len
            feed_dict = self.feed_data(batch_ids, batch_mask, batch_segment, batch_label, 1.0)
            loss, acc = sess.run([self.model.cost_func,
                     self.model.accuracy], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / data_len, total_acc / data_len

    def test(self):
        """testing"""

        save_dir = os.path.join(self.output_dir, "checkpoints/textcnn")
        save_path = os.path.join(save_dir, 'best_validation')

        if not os.path.exists(save_dir):
            tf.logging.info("maybe you don't train")
            exit()

        tf.logging.info("*****************Loading testing data*****************")
        test_examples = TextProcessor().get_test_examples(self.data_dir)
        test_data = convert_examples_to_features(test_examples, self.label_list, self.model.sequence_length, self.tokenizer)

        input_ids, input_mask, segment_ids = [], [], []

        for features in test_data:
            input_ids.append(features['input_ids'])
            input_mask.append(features['input_mask'])
            segment_ids.append(features['segment_ids'])

        self.model.is_training = False
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path)

        tf.logging.info('Testing...')
        test_loss, test_accuracy = self.evaluate(session, test_data)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        tf.logging.info(msg.format(test_loss, test_accuracy))

        batch_size = self.batch_size
        data_len = len(test_data)
        num_batch = int((data_len - 1) / batch_size) + 1
        y_test_cls = [features['label_ids'] for features in test_data]
        y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {
                self.model.input_ids: np.array(input_ids[start_id:end_id]),
                self.model.input_mask: np.array(input_mask[start_id:end_id]),
                self.model.segment_ids: np.array(segment_ids[start_id:end_id]),
                self.model.keep_prob: 1.0,
            }
            y_pred_cls[start_id:end_id] = session.run(self.model.y_pred_cls, feed_dict=feed_dict)

        # evaluate
        tf.logging.info("Precision, Recall and F1-Score...")
        tf.logging.info(metrics.classification_report(y_test_cls, y_pred_cls, target_names=self.label_list))

        tf.logging.info("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        tf.logging.info(cm)


if __name__ == '__main__':
    train = ModerTrain()
    train.train()
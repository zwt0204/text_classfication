# -*- encoding: utf-8 -*-
"""
@File    : xgboost.py
@Time    : 2019/12/20 10:33
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import gensim
import numpy as np
from sklearn.externals import joblib
from xgboost import XGBClassifier
import jieba
import json
from sklearn.metrics import precision_score, recall_score, f1_score

jieba.load_userdict('name_tokenizer.dat')
VECTOR_DIR = 'word2vec.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2

print('(1) load texts...')
train_f = open('train.txt', 'r', encoding='utf8')
test_f = open('test.txt', 'r', encoding='utf8')
stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf8').readlines()]
train_lines = train_f.readlines()
test_lines = test_f.readlines()
train_docs = []
train_labels = []
for line in train_lines:
    line = json.loads(line)
    data = jieba.lcut(str(line['text']))
    outstr = ''
    for word in data:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    train_docs.append(outstr)
    train_labels.append(line['id'])

test_docs = []
test_labels = []
for line in test_lines:
    line = json.loads(line)
    data = jieba.lcut(str(line['text']))
    outstr = ''
    for word in data:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    test_docs.append(outstr)
    test_labels.append(line['id'])

print('(2) doc to var...')

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)


def train():
    x_train = []
    for train_doc in train_docs:
        words = train_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train.append(vector)
    print('train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0])))
    y_train = train_labels
    print('(3) XGBOOST...')
    # 拟合XGBoost模型
    """
    max_depth=3,树的深度.调参：值越大，越容易过拟合；值越小，越容易欠拟合。典型值3-10
    learning_rate=0.1.学习率，控制每次迭代更新权重时的步长调参：值越小，训练越慢。典型值为0.01-0.2。
    n_estimators=100, 总共迭代的次数，即决策树的个数
    verbosity=1,  0-3
    silent=None,是否输出中间结果
    objective="binary:logistic", 目标函数
        回归任务
        reg:linear (默认)
        reg:logistic 
        二分类
        binary:logistic     概率 
        binary：logitraw   类别
        多分类
        multi：softmax  num_class=n   返回类别
        multi：softprob   num_class=n  返回概率
    rank:pairwise 
    booster='gbtree', gbtree 树模型做为基分类器（默认）、gbliner 线性模型做为基分类器
    n_jobs=1,  Number of parallel threads used to run xgboost
    nthread=None,  nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
    gamma=0,  惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    min_child_weight=1,
        含义：默认值为1,。
        调参：值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。
    max_delta_step=0,  Maximum delta step we allow each tree’s weight estimation to be
    subsample=1,
        含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。
        调参：防止overfitting。
    colsample_bytree=1,
        含义：训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
        调参：防止overfitting。
    colsample_bylevel=1,  Subsample ratio of columns for each level
    colsample_bynode=1,  Subsample ratio of columns for each split
    reg_alpha=0,  L1 regularization term on weights
    reg_lambda=1, L2 regularization term on weights
    scale_pos_weight=1,
        正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10。
    base_score=0.5,  The initial prediction score of all instances, global bias.
    random_state=0   Random number seed. (replaces seed)
    """
    model = XGBClassifier()
    model.fit(np.array(x_train), y_train)
    joblib.dump(model, "model.m")


def test():
    x_test = []
    for test_doc in test_docs:
        words = test_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_test.append(vector)
    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))
    y_test = test_labels
    print('(3) XGBOOST...')
    xgboostclf = joblib.load("model.m")
    preds = xgboostclf.predict(np.array(x_test))
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
        else:
            print(int(pred), '>>>>', test_docs[i].replace(' ', ''), '>>>>', y_test[i])
    print('precision_score:' + str(float(num) / len(preds)))

    precision = precision_score(y_test, preds)
    print(precision)
    recall = recall_score(y_test, preds)
    print(recall)
    f1 = f1_score(y_test, preds)
    print(f1)


def predict():
    x_test = []
    pre_items = ["我要上厕所"]
    for test_doc in pre_items:
        words = jieba.lcut(test_doc)
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_test.append(vector)
    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))
    print('(3) XGBOOST...')
    xgboostclf = joblib.load("model.m")
    preds = xgboostclf.predict_proba(x_test)
    """
    直接输出类别： predict(x_test)
    输出概率： predict_proba(x_test)
    """
    preds = preds.tolist()
    for i, j in zip(preds, pre_items):
        print(j, '>>>>', np.argmax(i), '>>>>', max(i))


if __name__ == '__main__':
    train()
    # test()
    # predict()

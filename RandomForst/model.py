# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2020/3/13 18:11
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import gensim
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import jieba
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# jieba.load_userdict('name_tokenizer.dat')
VECTOR_DIR = 'D:\mygit\\tf1.0\data\\raw_data\\100w.bin'

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2

print('(1) load texts...')
train_f = open('D:\mygit\\text_classfication\data\\train.txt', 'r', encoding='utf8')
test_f = open('D:\mygit\\text_classfication\data\\test.txt', 'r', encoding='utf8')
stopwords = [line.strip() for line in open('D:\mygit\\text_classfication\data\\stopwords.txt', 'r', encoding='utf8').readlines()]
train_lines = train_f.readlines()
test_lines = test_f.readlines()
train_docs = []
train_labels = []
for line in train_lines:
    line = json.loads(line)
    data = jieba.lcut(str(line['data']))
    outstr = ''
    for word in data:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    train_docs.append(outstr)
    train_labels.append(line['label'])

test_docs = []
test_labels = []
for line in test_lines:
    line = json.loads(line)
    data = jieba.lcut(str(line['data']))
    outstr = ''
    for word in data:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ' '
    test_docs.append(outstr)
    test_labels.append(line['label'])

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
    print('(3) RandomForestClassifier...')
    # 拟合模型
    """
    """
    model = RandomForestClassifier()
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
    print('(3) RandomForst...')
    RandomForestClassifierost = joblib.load("model.m")
    preds = RandomForestClassifierost.predict(np.array(x_test))
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
    print('(3) RandomForestClassifierost...')
    RandomForestClassifierost = joblib.load("model.m")
    preds = RandomForestClassifierost.predict_proba(x_test)
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




# -*- encoding: utf-8 -*-
"""
@File    : svm.py
@Time    : 2019/12/20 13:00
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import gensim
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
import json
import jieba
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
    print('(3) SVM...')
    """
    # C 表示对错误样本的惩罚，默认为1.惩罚大准确率高泛化性能低。
    # kernel核函数:‘linear’:线性核函数、‘poly’：多项式核函数、‘rbf’：径像核函数 / 高斯核、‘sigmod’:sigmod核函数、‘precomputed’:核矩阵
    # degree:int型参数 默认为3。这个参数只对多项式核函数有用，是指多项式核函数的阶数n
    # gamma：float参数 默认为auto。核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features.
    # coef0:float参数 默认为0.0。核函数中的独立项，只有对‘poly’和‘sigmod’核函数有用，是指其中的参数c
    # probability：bool参数 默认为False。是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢。
    # shrinking：bool参数 默认为True。是否采用启发式收缩方式
    # tol: float参数  默认为1e^-3。svm停止训练的误差精度
    # cache_size：float参数 默认为200。指定训练所需要的内存，以MB为单位，默认为200MB。
    # class_weight：字典类型或者‘balance’字符串。默认为None。
        # 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.
        # 如果给定参数‘balanced’，则使用y的值自动调整与输入数据中的类频率成反比的权重。
    verbose ：bool参数 默认为False.
        是否启用详细输出。 此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
    max_iter ：int参数 默认为-1. 最大迭代次数，如果为-1，表示不限制
    random_state：int型参数 默认为None.伪随机数发生器的种子,在混洗数据时用于概率估计。
    """

    svclf = SVC(C=1, kernel='rbf', probability=True, tol=1e-5, class_weight='balanced')
    svclf.fit(x_train, y_train)
    joblib.dump(svclf, "model.m")


def test1():
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
    print('(3) SVM...')
    svclf = joblib.load("model.m")
    preds = svclf.predict(x_test)
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
    pre_items = ["请问厕所怎么走"]

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
    print('(3) SVM...')
    svmclf = joblib.load("model.m")
    preds = svmclf.predict_proba(x_test)
    """
    直接输出类别： predict(x_test)
    输出概率： predict_proba(x_test)
    """


if __name__ == '__main__':
    train()
    # test()
    # predict()

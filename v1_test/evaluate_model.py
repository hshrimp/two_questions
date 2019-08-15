#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/14 下午2:45
"""

from keras.models import load_model
import numpy as np
from v1_test.config import *
from sklearn.metrics import accuracy_score
from v1_test.tokenizer_v1 import tokenizer
from v1_test.utils import seq_padding, get_data
import pandas as pd


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.test_y = []

    def __len__(self):
        return self.steps

    @staticmethod
    def generator_y(length1, length2):
        li = [0] * length1
        li.append(1)
        li.extend([0] * (length2 - 1))
        return li[:maxlen]

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                first = self.data[i]
                second = self.data[-i]
                text = (first + second)[:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = self.generator_y(len(first), len(second))
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    # Y = seq_padding(Y)
                    self.test_y.extend(Y)
                    yield [X1, X2]
                    [X1, X2, Y] = [], [], []


def generator_y(length1, length2):
    li = [0] * length1
    li.append(1)
    li.extend([0] * (length2 - 1))
    return li[:maxlen]


def generator_test_data(data):
    idxs = list(range(len(data)))
    np.random.shuffle(idxs)
    X1, X2, Y, texts = [], [], [], []
    for i in idxs:
        first = data[i]
        second = data[-i]
        text = (first + second)[:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        # y = generator_y(len(first), len(second))
        X1.append(x1)
        X2.append(x2)
        Y.append(len(first))
        texts.append(text)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    return X1, X2, Y, texts


def write_to_csv(texts, Y, pred):
    table = pd.DataFrame()
    table['texts'] = texts
    table['text1'] = [t[:point] for t, point in zip(texts, pred)]
    table['text2'] = [t[point:] for t, point in zip(texts, pred)]
    table['Y'] = Y
    table["pred"] = pred

    table.to_csv('result.csv', index=None)


if __name__ == '__main__':
    data_path = '/home/wushaohong/Downloads/lcqmc中文问句相似对/test.tsv'
    valid_data = get_data(data_path)
    X1, X2, Y, texts = generator_test_data(valid_data)
    model = load_model('model_v1.h5', custom_objects=custom_dict)
    pred = model.predict([X1, X2], batch_size=64)
    pred = np.argmax(pred, axis=1)
    print(accuracy_score(Y, pred))
    X1 = list(X1)
    pred = list(pred)
    write_to_csv(texts, Y, pred)

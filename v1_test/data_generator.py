#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/14 上午11:24
"""
from v1_test.config import maxlen
import numpy as np
from v1_test.utils import seq_padding


class DataGenerator:
    def __init__(self, data, tokenizer, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

        self.tokenizer = tokenizer

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
                x1, x2 = self.tokenizer.encode(first=text)
                y = self.generator_y(len(first), len(second))
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

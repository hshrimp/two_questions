#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/14 下午3:50
"""
import pandas as pd
from v1_test.config import punctuation, maxlen
import numpy as np


def get_data(path):
    table = pd.read_csv(path, sep='\t', header=0)

    data = []

    def remove_punctuation(text):
        if text[-1] in punctuation:
            text = text[:-1]
        return text

    for i in range(len(table)):
        text = remove_punctuation(table.text_a[i])
        data.append(text)
        text = remove_punctuation(table.text_b[i])
        data.append(text)
    return data


def seq_padding(X, padding=0):
    return np.array([
        np.concatenate([x, [padding] * (maxlen - len(x))]) for x in X])

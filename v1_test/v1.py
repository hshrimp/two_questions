#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/14 上午11:20
"""

from keras_bert import load_trained_model_from_checkpoint
from v1_test.config import *
from v1_test.tokenizer_v1 import tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from v1_test.utils import get_data
from v1_test.data_generator import DataGenerator


def split_data(data):
    # 按照9:1的比例划分训练集和验证集
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
    return train_data, valid_data


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    bert_model.summary()
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0], name='Lambda')(x)
    p = Dense(maxlen, activation='softmax', name='out_Dense')(x)

    model = Model([x1_in, x2_in], p)
    return model


def main(path):
    data = get_data(path)
    train_data, valid_data = split_data(data)
    train_D = DataGenerator(train_data, tokenizer)
    valid_D = DataGenerator(valid_data, tokenizer)

    model = get_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )

    model.summary()

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=1,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )
    model.save('model_v1_2.h5')


if __name__ == '__main__':
    data_path = '/home/wushaohong/Downloads/lcqmc中文问句相似对/dev.tsv'
    main(data_path)

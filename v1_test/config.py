#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/14 下午3:14
"""
from keras_bert.layers import TokenEmbedding
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from keras_bert import gelu_tensorflow

punctuation = ',.?!，。？！'
data_dir = '/home/wushaohong/PycharmProjects/bert-chinese-ner/checkpoint/'

maxlen = 72
config_path = data_dir + 'bert_config.json'
checkpoint_path = data_dir + 'bert_model.ckpt'
dict_path = data_dir + 'vocab.txt'

custom_dict = {'TokenEmbedding': TokenEmbedding, 'PositionEmbedding': PositionEmbedding,
                   'LayerNormalization': LayerNormalization, 'MultiHeadAttention': MultiHeadAttention,
                   'FeedForward': FeedForward, 'gelu_tensorflow': gelu_tensorflow}
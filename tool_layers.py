#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *

"""自定义工具层
embedding, crossDot, crossMulti, one-hot, dice ,etc
"""


class HashBucketsEmbedding(Layer):
    def __init__(self,
                 num_buckets,
                 emb_size,
                 **kwargs):
        super(HashBucketsEmbedding, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.emb_size = emb_size

    def build(self, input_shape):
        super(HashBucketsEmbedding, self).build(input_shape)
        self.embedding_layer = Embedding(input_dim=self.num_buckets + 1,
                                         output_dim=self.emb_size,
                                         name='embedding')

    def call(self, input):
        emb_input = []
        for i in range(input.shape[1]):
            x = tf.as_string(tf.slice(input, [0, i], [-1, 1]))
            emb_input.append(x)
        emb_input = tf.concat(emb_input, 1)
        emb_input = tf.strings.to_hash_bucket_strong(emb_input, self.num_buckets, [1, 2])  # hash
        out = self.embedding_layer(emb_input)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_buckets": self.num_buckets,
            "emb_size": self.emb_size,
        })
        return config


# tensorflow 内置字典查询
class VocabLayer(Layer):
    def __init__(self, vocab_path, vocab_name, in_type=tf.int64, out_type=tf.int64, sep='\t', **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.vocab_path = vocab_path
        self.vocab_name = vocab_name
        self.in_type = in_type
        self.out_type = out_type
        self.sep = sep

    def build(self, input_shape):
        super(VocabLayer, self).build(input_shape)
        if os.path.isdir(self.vocab_path):
            tmp = []
            for fp in os.listdir(self.vocab_path):
                f = pd.read_csv(os.path.join(self.vocab_path, fp), sep=self.sep, names=['key', 'value'])
                tmp.append(f)
            self.vocab = pd.concat(tmp, axis=0, ignore_index=True)
        else:
            self.vocab = pd.read_csv(self.vocab_path, names=['key', 'value'], sep=self.sep, header=None)

        # self.vocab['key'] = self.vocab['key'].apply(lambda x: int(x))
        self.table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(self.vocab['key'].values, dtype=self.in_type),
            values=tf.constant(self.vocab['value'].values, dtype=self.out_type), ),
            default_value=tf.constant(0, dtype=self.out_type), name=self.vocab_name)

    def call(self, input):
        token_ids = self.table.lookup(input)
        return token_ids

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_path": self.vocab_path,
            "vocab_name": self.vocab_name,
            "in_type": self.in_type,
            "out_type": self.out_type,
            "sep": self.sep,
        })
        return config


class L2_norm_layer(Layer):
    def __init__(self, axis, **kwargs):
        super(L2_norm_layer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)


class Power_layer(Layer):
    def __init__(self, y, **kwargs):
        super(Power_layer, self).__init__(**kwargs)
        self.y = tf.constant([y], dtype=tf.float32)

    def call(self, inputs):
        return tf.math.pow(inputs, self.y)


class CrossDotLayer(Layer):
    def __init__(self, axes, **kwargs):
        super(CrossDotLayer, self).__init__(**kwargs)
        self.axes = axes
        self.supports_masking = True

    def call(self, emb1, emb2, mask=None):
        return Dot(self.axes)([emb1, emb2])

    def compute_mask(self, inputs, mask=None):
        return None   # mask 到该层结束，不向下传递


class CrossMultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(CrossMultiplyLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, emb1, emb2, mask=None):
        return Multiply()([emb1, emb2])

    def compute_mask(self, inputs, mask=None):
        return None


class OneHotEncodingLayer(Layer):
    def __init__(self, num_classes, **kwargs):
        super(OneHotEncodingLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs):
        return tf.one_hot(inputs, self.num_classes)


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x


class MySoftmax(Layer):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def call(self, inputs):
        pos_out = inputs['pos']
        neg1_out = inputs['neg1']
        neg2_out = inputs['neg2']
        neg3_out = inputs['neg3']
        neg4_out = inputs['neg4']
        neg5_out = inputs['neg5']
        sum_e_xj = tf.exp(pos_out) + tf.exp(neg1_out) + tf.exp(neg2_out) + tf.exp(neg3_out) + tf.exp(neg4_out) + tf.exp(neg5_out)
        return concatenate([tf.exp(pos_out) / sum_e_xj,
                            tf.exp(neg1_out) / sum_e_xj,
                            tf.exp(neg2_out) / sum_e_xj,
                            tf.exp(neg3_out) / sum_e_xj,
                            tf.exp(neg4_out) / sum_e_xj,
                            tf.exp(neg5_out) / sum_e_xj,
                            ], axis=-1, name='softmax_pred')

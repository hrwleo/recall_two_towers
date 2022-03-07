#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *

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


# tensorflow 内置字典查询
class VocabLayer(Layer):
    def __init__(self, vocab_path, vocab_name, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.vocab_path = vocab_path
        self.vocab_name = vocab_name

    def build(self, input_shape):
        super(VocabLayer, self).build(input_shape)
        self.vocab = pd.read_csv(self.vocab_path, names=['key', 'value'], sep='\t', header=None)
        self.vocab['key'] = self.vocab['key'].apply(lambda x: int(x))
        self.table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(self.vocab['key'].values, dtype=tf.int64),
            values=tf.constant(self.vocab['value'].values, dtype=tf.int64), ),
            default_value=tf.constant(0, dtype=tf.int64), name=self.vocab_name)

    def call(self, input):
        token_ids = self.table.lookup(input)
        return token_ids


class L2_norm_layer(Layer):
    def __init__(self, axis, **kwargs):
        super(L2_norm_layer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs)


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

    def call(self, emb1, emb2):
        return Dot(self.axes)([emb1, emb2])


class CrossMultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(CrossMultiplyLayer, self).__init__(**kwargs)

    def call(self, emb1, emb2):
        return Multiply()([emb1, emb2])


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

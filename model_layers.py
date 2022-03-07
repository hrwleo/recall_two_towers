#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28
# update: 添加注释，完善自定义层 by stefan 2022-03-02

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

"""自定义模型层
Tower, DNN, SENet, DIN-Attention, ResNet, FM, DCN, CIN ..etc
"""


class FMLayer(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
         without linear term and bias.
          Input shape
            - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
          Output shape
            - 2D tensor with shape: ``(batch_size, 1)``.
        usage: FMLayer()(tf.stack(cross_emb_list, axis=1, name='fm_inputs'))
    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        concated_embeds_value = inputs
        # 先求和再平方
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        # 先平方再求和
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class ResNetLayer(Layer):
    """残差网络，改写卷积为全连接层
        Input shape
            - 2D tensor with shape: ``(batch_size, input_dim)``.
        Output shape
            - 2D tensor with shape: ``(batch_size, units)``.
    """

    def __init__(self, hidden_units=None, **kwargs):
        super(ResNetLayer, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [256, 128, 64]
        self.hidden_units = hidden_units
        self.dense_layers = []
        self.layer_num = len(self.hidden_units)
        self.relu = ReLU()
        self.batch_norm = BatchNormalization()
        self.add = Add()

    def build(self, input_shape):
        super(ResNetLayer, self).build(input_shape)
        for i in range(self.layer_num):
            dense_layer = Dense(self.hidden_units[i], activation=None)
            self.dense_layers.append(dense_layer)
        self.down_sample = Dense(self.hidden_units[self.layer_num - 1], activation=None)  # 最后一层要维度一样，便于最后Add

    def call(self, inputs):
        identity = self.down_sample(inputs)

        net = inputs
        for i in range(self.layer_num):
            net = self.dense_layers[i](net)
            if i == 0:
                net = self.batch_norm(net)
            if i != self.layer_num - 1:
                net = self.relu(net)

        output = self.relu(self.add([net, identity]))
        return output


class Tower(Layer):
    def __init__(self,
                 layer_num,
                 layer_units,
                 activation,
                 **kwargs):
        super(Tower, self).__init__(**kwargs)
        self.tower_layers = []
        self.layer_num = layer_num
        self.layer_units = layer_units
        self.activation = activation

    def build(self, input_shape):
        super(Tower, self).build(input_shape)
        for i in range(self.layer_num):
            dense_layer = Dense(self.layer_units[i], activation=self.activation)
            self.tower_layers.append(dense_layer)

    def call(self, inputs):
        net = inputs
        for layer in self.tower_layers:
            net = layer(net)
            net = Dropout(0.5)(net)
        return net


class SENetLayer(Layer):
    def __init__(self, last_shape, reduction=4, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.reduction = reduction
        self.last_shape = last_shape
        self.excitation_layer = Dense(self.last_shape, activation=tf.keras.activations.hard_sigmoid)
        self.squeeze_layer = Dense(self.last_shape // self.reduction, activation='relu')

    def call(self, inputs):
        net = self.squeeze_layer(inputs)
        net = self.excitation_layer(net)
        return net  # senet层输出的特征裁判值


class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """
        Linear Layer
        Input:
            - feature_length: A scalar. The length of features.
            - w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        return result


class MyDense(Layer):
    def __init__(self, units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        super(MyDense, self).build(input_shape)  # 相当于设置self.build = True
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='w')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')

    def call(self, inputs):
        return tf.keras.activations.sigmoid(tf.matmul(inputs, self.w) + self.b)


class DNNLayer(Layer):
    def __init__(self, layer_units, **kwargs):
        super(DNNLayer, self).__init__(**kwargs)
        self.layer_units = layer_units
        self.batch_norm = BatchNormalization()

        self.dense_layers = []

    def build(self, input_shape):
        super(DNNLayer, self).build(input_shape)
        for i in range(len(self.layer_units)):
            dense_layer = Dense(self.layer_units[i], activation=None)
            self.dense_layers.append(dense_layer)

    def call(self, inputs, training=False):
        net = inputs
        for i in range(len(self.dense_layers)):
            net = self.dense_layers[i](net)
            if i == 0:
                net = self.batch_norm(net)  # batch_norm加在第一层的输入的线性变换后，激活函数前
            net = ReLU()(net)
            if training:
                net = Dropout(0.3)(net)
        return net


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='relu'):
        """
            Input shape
                - query: 2D tensor with shape: ``(batch_size, input_dim)``.
                - key: 3D tensor with shape: ``(batch_size, seq_len, input_dim)``.
                - value: 3D tensor with shape: ``(batch_size, seq_len, input_dim)``.
            Output shape
                - 2D tensor with shape: ``(batch_size, input_dim)``.
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = []
        self.att_hidden_units = att_hidden_units
        self.activation = activation
        self.att_final_dense = Dense(1)

    def build(self, input_shape):
        super(Attention_Layer, self).build(input_shape)
        for i in range(len(self.att_hidden_units)):
            self.att_dense.append(Dense(self.att_hidden_units[i], activation=self.activation))

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        q, k, v = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class ActivationSumPoolingFromDIN(Layer):
    def __init__(self, att_hidden_units=[64, 32], att_activation='relu'):
        """
        用户行为序列对候选集做atten，然后sum pooling
        """
        super(ActivationSumPoolingFromDIN, self).__init__()

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)

    def call(self, inputs):
        seq_embed, item_embed = inputs
        user_interest_sum_pool = self.attention_layer([item_embed, seq_embed, seq_embed])

        # concat user_info(att hist), cadidate item embedding
        info_all = tf.concat([user_interest_sum_pool, item_embed], axis=-1)
        info_all = self.bn(info_all)
        return info_all


class DeepCrossLayer(Layer):
    def __init__(self, layer_num, embed_dim, **kwargs):
        """
            DCN Model implements
            usage: DeepCrossLayer(2, item_feature.shape[-1], name="deep_cross_features")(item_feature)
        """
        super(DeepCrossLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.embed_dim = embed_dim
        self.w = tf.Variable(lambda: tf.random.truncated_normal(shape=(self.embed_dim,), stddev=0.01))
        self.b = tf.Variable(lambda: tf.zeros(shape=(embed_dim,)))

    def cross_layer(self, inputs):
        x0, xl = inputs
        # feature crossing
        x1_T = tf.reshape(xl, [-1, 1, self.embed_dim])
        x_lw = tf.tensordot(x1_T, self.w, axes=1)
        cross = x0 * x_lw
        return cross + self.b + xl

    def call(self, inputs):
        xl = inputs
        for i in range(self.layer_num):
            xl = self.cross_layer([inputs, xl])

        return xl


class CINLayer(Layer):
    def __init__(self, cin_size=[64, 64], l2_reg=1e-4, **kwargs):
        """CIN Model implements
        ** only for sparse feature **

            Input
                - cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
                - l2_reg: A scalar. L2 regularization.
                - inputs tensor 3-D (batch_size, field_nums, emb_sizes)
            usage: CINLayer()(tf.stack([item_shangquan_emb, item_comm_emb], axis=1), name='cin_features')
        """
        super(CINLayer, self).__init__(**kwargs)
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  # dim * (None, filed_nums[i], 1)

            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result, axis=-1)  # (None, dim)

        return result


def parallel_layer(tower_num_layer, tower_num_layer_units, mlp_inputs, fm_inputs, dcn_inputs, cin_inputs):
    mlp_features = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                         activation=tf.nn.leaky_relu)(mlp_inputs)
    fm_features = FMLayer()(fm_inputs)
    dcn_features = DeepCrossLayer(2, dcn_inputs.shape[-1])(dcn_inputs)
    cin_features = CINLayer()(cin_inputs)

    # concat dnn_out and dcn_out
    mlp_dcn_features = concatenate([mlp_features, dcn_features], axis=-1)

    return mlp_dcn_features, fm_features, cin_features

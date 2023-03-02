#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28
# update: 添加注释，完善自定义层 by stefan 2022-03-02
import numpy as np
import tensorflow as tf
from keras import initializers, regularizers, constraints
from keras.backend import expand_dims, repeat_elements, sum
from keras.layers import *
from keras.regularizers import l2

from layers.tool_layers import L2_norm_layer

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

    def call(self, inputs, **kwargs):
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

    def call(self, inputs, **kwargs):
        net = inputs
        for layer in self.tower_layers:
            net = layer(net)
        net = Dropout(0.3)(net)
        return net


class SENetLayer(Layer):
    def __init__(self, last_shape, reduction=4, **kwargs):
        super(SENetLayer, self).__init__(**kwargs)
        self.reduction = reduction
        self.last_shape = last_shape
        self.excitation_layer = Dense(self.last_shape, activation=tf.keras.activations.hard_sigmoid)
        self.squeeze_layer = Dense(self.last_shape // self.reduction, activation='relu')

    def call(self, inputs, **kwargs):
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

    def call(self, inputs, **kwargs):
        return tf.keras.activations.sigmoid(tf.matmul(inputs, self.w) + self.b)


class DNNLayer(Layer):
    def __init__(self, layer_units, dropout_rate=0.3, **kwargs):
        super(DNNLayer, self).__init__(**kwargs)
        self.layer_units = layer_units
        self.batch_norm = BatchNormalization()
        self.dropout_rate = dropout_rate
        self.dense_layers = []

    def build(self, input_shape):
        super(DNNLayer, self).build(input_shape)
        for i in range(len(self.layer_units)):
            dense_layer = Dense(self.layer_units[i], activation='relu')
            self.dense_layers.append(dense_layer)

    def call(self, inputs, **kwargs):
        net = inputs
        for i in range(len(self.dense_layers)):
            net = self.dense_layers[i](net)
            if i == 0:
                net = self.batch_norm(net)  # batch_norm加在第一层的输入的线性变换后，激活函数(Relu)之后
        net = Dropout(self.dropout_rate)(net)
        return net


class UserRepresentationLayer(Layer):
    def __init__(self, **kwargs):
        super(UserRepresentationLayer, self).__init__(**kwargs)
        self.ActivationSumPoolingFromDIN = ActivationSumPoolingFromDIN()

    def call(self, inputs, **kwargs):
        em, eu, Xu = inputs
        ru_ = self.ActivationSumPoolingFromDIN([Xu, em])

        # ru: user representation
        ru = concatenate([ru_, eu], axis=-1)
        return ru


class UserMatchLayer(Layer):
    def __init__(self, **kwargs):
        super(UserMatchLayer, self).__init__(**kwargs)
        self.l2_norm_layer = L2_norm_layer(axis=-1)

    def relavant_unit(self, ru, r_ul):
        ru_norm = self.l2_norm_layer(ru)
        r_ul_norm = self.l2_norm_layer(r_ul)
        a_l = tf.reduce_sum(tf.multiply(ru_norm, r_ul_norm), axis=1, keepdims=True)

        relavant = {'relavant': tf.multiply(a_l, r_ul),
                  'a_l': a_l
                  }
        return relavant

    def call(self, inputs, **kwargs):
        ru, ru1, ru2, ru3 = inputs
        ru_u1 = self.relavant_unit(ru, ru1)
        ru_u2 = self.relavant_unit(ru, ru2)
        ru_u3 = self.relavant_unit(ru, ru3)

        result = {'Su': ru_u1['relavant'] + ru_u2['relavant'] + ru_u3['relavant'],
                  'Ru': ru_u1['a_l'] + ru_u2['a_l'] + ru_u3['a_l']
                  }
        return result


class TextCNNLayer(Layer):
    def __init__(self, filters, kernel_size, hidden_units, **kwargs):
        super(TextCNNLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.convs = []
        self.max_pools = []
        for i in range(len(self.kernel_size)):
            self.kernel_size[i] = int(self.kernel_size[i]) if not isinstance(self.kernel_size[i], int) else self.kernel_size[i]
            conv_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size[i], padding='same', strides=1, activation='relu')
            max_pool = MaxPooling1D(pool_size=self.kernel_size[i], padding='same')
            self.convs.append(conv_layer)
            self.max_pools.append(max_pool)
        self.batch_norm = BatchNormalization()
        self.dense_layer = Dense(self.hidden_units, activation='relu')

    def call(self, inputs, **kwargs):
        cnn_i = []
        for i in range(len(self.convs)):
            x = self.convs[i](inputs)  # 每次对inputs做不同尺度的卷积
            x = self.max_pools[i](x)
            cnn_i.append(Flatten()(x))

        cnn = concatenate(cnn_i, axis=-1)

        drop = Dropout(0.3)(cnn)
        out = self.dense_layer(drop)
        return out


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
        self.supports_masking = True

    def build(self, input_shape):
        super(Attention_Layer, self).build(input_shape)
        for i in range(len(self.att_hidden_units)):
            self.att_dense.append(Dense(self.att_hidden_units[i], activation=self.activation))

    def call(self, inputs, mask=None, **kwargs):
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

        if mask:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)  填充 -inf
            outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class SelfAttention_Layer(Layer):
    def __init__(self):
        super(SelfAttention_Layer, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(shape=[self.dim, self.dim], name='weight',
            initializer='random_uniform')

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs
        # pos encoding
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        if mask:
            mask = tf.tile(tf.expand_dims(mask, 1), [1, q.shape[1], 1])  # (None, seq_len, seq_len)
            paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class BiLSTM_Attention_Layer(Layer):
    def __init__(self, lstm_units=None, **kwargs):
        super(BiLSTM_Attention_Layer, self).__init__(**kwargs)
        self.lstm_units = lstm_units
        self.bi_lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')
        self.bi_lstm2 = Bidirectional(LSTM(lstm_units))

    def call(self, inputs, **kwargs):
        inputs = Reshape((-1, inputs.shape[1]))(inputs)
        bilstm_out1 = self.bi_lstm1(inputs)
        bilstm_out2 = self.bi_lstm2(bilstm_out1)
        #att_out = self.self_att(bilstm_out2)
        return bilstm_out2


class ActivationSumPoolingFromDIN(Layer):
    def __init__(self, att_hidden_units=[64, 32], att_activation='relu'):
        """
        用户行为序列对候选集做atten，然后sum pooling
        """
        super(ActivationSumPoolingFromDIN, self).__init__()

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)

    def call(self, inputs, **kwargs):
        seq_embed, item_embed = inputs
        user_interest_sum_pool = self.attention_layer([item_embed, seq_embed, seq_embed])

        # concat user_info(att hist), cadidate item embedding
        info_all = tf.concat([user_interest_sum_pool, item_embed], axis=-1)
        info_all = self.bn(info_all)
        return info_all


class MultiHeadSelfAttention(Layer):
    def __init__(self, num_units, num_heads=8, dropout_rate=0, **kwargs):
        """
            Applies multi-head attention.
                Args:
                  queries: A 3d tensor with shape of [N, T_q, C_q].
                  keys: A 3d tensor with shape of [N, T_k, C_k].
                  values: A 3d tensor with shape of [N, T_v, C_v]
                  num_units: A scalar. Attention size.
                  dropout_rate: A floating point number.
                  num_heads: An int. Number of heads.
                Returns
                  A 3d tensor with shape of (N, T_q, C)
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dense_q = Dense(units=self.num_units, use_bias=False, activation='relu')
        self.dense_k = Dense(units=self.num_units, use_bias=False, activation='relu')
        self.dense_v = Dense(units=self.num_units, use_bias=False, activation='relu')

    def call(self, inputs, **kwargs):
        queries, keys, values = inputs
        Q = self.dense_q(queries)
        K = self.dense_k(keys)
        V = self.dense_v(values)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Dropouts
        outputs = Dropout(self.dropout_rate)(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


class DeepCrossLayer(Layer):
    def __init__(self, layer_num, embed_dim, output_dim=0, **kwargs):
        """
            DCN Model implements
            usage: DeepCrossLayer(2, item_feature.shape[-1], name="deep_cross_features")(item_feature)
        """
        super(DeepCrossLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.embed_dim = embed_dim

        self.w = []
        self.b = []
        for i in range(self.layer_num):
            self.w.append(tf.Variable(lambda: tf.random.truncated_normal(shape=(self.embed_dim,), stddev=0.01)))
            self.b.append(tf.Variable(lambda: tf.zeros(shape=(embed_dim,))))

        self.output_dim = output_dim
        self.dense = Dense(units=self.output_dim, use_bias=False)

    def cross_layer(self, inputs, i):
        x0, xl = inputs
        # feature crossing
        x1_T = tf.reshape(xl, [-1, 1, self.embed_dim])
        x_lw = tf.tensordot(x1_T, self.w[i], axes=1)
        cross = x0 * x_lw
        return cross + self.b[i] + xl

    def call(self, inputs, **kwargs):
        xl = inputs
        for i in range(self.layer_num):
            xl = self.cross_layer([inputs, xl], i)
        if self.output_dim > 0:
            xl = self.dense(xl)
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


def parallel_layer(num_layer, layer_units, mlp_inputs, fm_inputs, dcn_inputs, cin_inputs):
    """
        腾讯信息流推荐排序中的并联双塔CTR结构
        复现参考 add by stefan
    """
    mlp_features = Tower(layer_num=num_layer, layer_units=layer_units,
                         activation=tf.nn.leaky_relu)(mlp_inputs)
    fm_features = FMLayer()(fm_inputs)
    dcn_features = DeepCrossLayer(2, dcn_inputs.shape[-1], int(layer_units[-1]))(dcn_inputs)
    cin_features = CINLayer(cin_size=[32, 32])(cin_inputs)

    # concat dnn_out and dcn_out
    mlp_dcn_features = concatenate([mlp_features, dcn_features], axis=-1)

    return mlp_dcn_features, fm_features, cin_features


class GlobalAveragePooling1DSef(Layer):
    def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
        super(GlobalAveragePooling1DSef, self).__init__(**kwargs)
        self.data_format = data_format
        self.supports_masking = True
        self.keepdims = keepdims

    def call(self, inputs, mask=None, **kwargs):
        steps_axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == 'channels_last' else 1)
            inputs *= mask
            return tf.reduce_sum(
                inputs, axis=steps_axis,
                keepdims=self.keepdims) / tf.maximum(1.0, tf.reduce_sum(
                mask, axis=steps_axis, keepdims=self.keepdims))
        else:
            return tf.reduce_mean(inputs, axis=steps_axis, keepdims=self.keepdims)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "data_format": self.data_format,
            "keepdims": self.keepdims,
        })
        return config


class MMoE(Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        # self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(Dense(self.units, activation=self.expert_activation,
                                            use_bias=self.use_expert_bias,
                                            kernel_initializer=self.expert_kernel_initializer,
                                            kernel_regularizer=self.expert_kernel_regularizer,
                                            bias_regularizer=self.expert_bias_regularizer,
                                            activity_regularizer=None,
                                            kernel_constraint=self.expert_kernel_constraint,
                                            bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(Dense(self.num_experts, activation=self.gate_activation,
                                          use_bias=self.use_gate_bias,
                                          kernel_initializer=self.gate_kernel_initializer,
                                          kernel_regularizer=self.gate_kernel_regularizer,
                                          bias_regularizer=self.gate_bias_regularizer,
                                          activity_regularizer=None,
                                          kernel_constraint=self.gate_kernel_constraint,
                                          bias_constraint=self.gate_bias_constraint))

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        # assert input_shape is not None and len(input_shape) >= 2

        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs, 2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            aa = repeat_elements(expanded_gate_output, self.units, axis=1)
            weighted_expert_output = expert_outputs * aa
            bb = sum(weighted_expert_output, axis=2)
            final_outputs.append(bb)
        # 返回的矩阵维度 num_tasks * batch * units

        return final_outputs

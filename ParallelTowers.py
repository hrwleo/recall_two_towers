#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-03-03

"""
参考腾讯并联双塔模型架构 尝试复现模型

主要创新思路在于：
1、尝试通过＂并联＂多个双塔结构（MLP、DCN、FM、FFM、CIN）增加双塔模型的＂宽度＂来缓解双塔内积的瓶颈从而提升效果；
2、对＂并联＂的多个双塔引入 LR 进行带权融合，LR 权重最终融入到 userembedding 中，使得最终的模型仍然保持的内积形式。
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from layers.tool_layers import *
from layers.model_layers import MyDense, parallel_layer


# build model
def parallel_towers(temperature, city_dict, shangquan_dict, comm_dict, price_dict, area_dict, tower_num_layer,
                    tower_num_layer_units):
    # **********************************  输入层 **********************************#
    # define input
    

    # common emb  （目前只有一个城市的数据，适当调整dim)


    # user feature


    # item features
    

    # **********************************  表示层 **********************************#
    user_mlp_inputs = concatenate([...],
                                  axis=-1, name='user_mlp_inputs')
    user_fm_inputs = tf.stack([...],
                              axis=1, name='user_fm_inputs')
    user_dcn_inputs = concatenate([...], axis=-1, name="user_dcn_inputs")
    user_cin_inputs = tf.stack([...], axis=1, name="user_cin_inputs")

    # 计算user_tower的并联输出
    user_mlp_dcn_out, user_fm_out, user_cin_out = parallel_layer(tower_num_layer, tower_num_layer_units,
                                                                 user_mlp_inputs, user_fm_inputs, user_dcn_inputs,
                                                                 user_cin_inputs)

    item_mlp_inputs = concatenate([...],
                                  axis=-1, name="item_mlp_inputs")
    item_fm_inputs = tf.stack([...], axis=1,
                              name="item_fm_inputs")

    item_dcn_inputs = concatenate([...], axis=-1, name="item_dcn_inputs")
    item_cin_inputs = tf.stack([...], axis=1, name="item_cin_inputs")

    # 计算item_tower的并联输出
    item_mlp_dcn_out, item_fm_out, item_cin_out = parallel_layer(tower_num_layer, tower_num_layer_units,
                                                                 item_mlp_inputs, item_fm_inputs, item_dcn_inputs,
                                                                 item_cin_inputs)

    # **********************************  匹配层 **********************************#
    # 按照不同并联模型分别进行 hadamard 积， 在顶层做两侧特征的交互
    user_item_mlp_dcn_hdm = tf.multiply(user_mlp_dcn_out, item_mlp_dcn_out)
    user_item_fm_hdm = tf.multiply(user_fm_out, item_fm_out)
    user_item_cin_hdm = tf.multiply(user_cin_out, item_cin_out)

    # 使用LR学习＂并联＂的多个双塔的权重
    my_dense = MyDense(1)
    concat_inputs = concatenate([user_item_mlp_dcn_hdm, user_item_fm_hdm, user_item_cin_hdm], axis=-1)
    out = my_dense(concat_inputs)
    lr_weights = my_dense.weights[0]
    lr_weights = tf.reshape(lr_weights, [1, lr_weights.shape[0]])

    user_input = [...]
    item_input = [...]

    # 获取双塔各自的emb输出
    user_parallel_out = concatenate([user_mlp_dcn_out, user_fm_out, user_cin_out], axis=-1, name="user_tower_out")
    item_parallel_out = concatenate([item_mlp_dcn_out, item_fm_out, item_cin_out], axis=-1, name="item_tower")

    user_output = tf.multiply(user_parallel_out, lr_weights, name="user_tower")  # 预先融合LR的权重进user embedding
    item_output = item_parallel_out

    user_model = Model(inputs=user_input, outputs=user_output)
    item_model = Model(inputs=item_input, outputs=item_output)
    all_model = Model(inputs=user_input + item_input, outputs=out)  # 模型评估以out为评估对象

    all_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-3),
        metrics=[tf.keras.metrics.AUC()]
    )

    all_model.summary()

    return all_model, user_model, item_model

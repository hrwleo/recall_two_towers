#!/usr/bin/env python
# coding: utf-8
# author: stefan 2023-03-01

"""
基于Learning to Rank的双塔召回
Pairwise-双塔模型
"""
from keras import Model
from keras.layers import Reshape, concatenate, Dense

from layers.metrics import softmaxloss, myLtrAcc
from layers.model_layers import Tower
from layers.tool_layers import L2_norm_layer, MySoftmax
from recall.LTR.build_feature import build_feature_column
import tensorflow as tf

def build_ltr_tower(temperature, city_dict, shangquan_dict, comm_dict, price_dict, area_dict, tower_num_layer,
                   tower_num_layer_units):
    # 输入、特征
    feature_columns = build_feature_column(city_dict, shangquan_dict, comm_dict, price_dict, area_dict)

    # user Tower
    # 计算user tower的输出
    user_tower_out = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                           activation=tf.nn.leaky_relu, name='user_tower')(feature_columns['user_ft'])
    user_tower_out_norm = L2_norm_layer(axis=-1, name='user_tower_norm')(user_tower_out)

    # item tower (参数共享) 初始化
    item_model = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                       activation=tf.nn.leaky_relu, name='item_tower')

    # pos house tower的输出
    house_pos_out = item_model(feature_columns['pos_house_ft'])
    house_pos_out_norm = L2_norm_layer(axis=-1, name='pos_tower')(house_pos_out)

    # neg1 house tower的输出
    house_neg_1_out = item_model(feature_columns['neg_house_1_ft'])
    house_neg_1_out_norm = L2_norm_layer(axis=-1, name='neg_tower_1')(house_neg_1_out)

    # neg2 house tower的输出
    house_neg_2_out = item_model(feature_columns['neg_house_2_ft'])
    house_neg_2_out_norm = L2_norm_layer(axis=-1, name='neg_tower_2')(house_neg_2_out)

    # neg3 house tower的输出
    house_neg_3_out = item_model(feature_columns['neg_house_3_ft'])
    house_neg_3_out_norm = L2_norm_layer(axis=-1, name='neg_tower_3')(house_neg_3_out)

    # neg4 house tower的输出
    house_neg_4_out = item_model(feature_columns['neg_house_4_ft'])
    house_neg_4_out_norm = L2_norm_layer(axis=-1, name='neg_tower_4')(house_neg_4_out)

    # neg5 house tower的输出
    house_neg_5_out = item_model(feature_columns['neg_house_5_ft'])
    house_neg_5_out_norm = L2_norm_layer(axis=-1, name='neg_tower_5')(house_neg_5_out)

    # user x pos_house inner product
    user_pos_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_pos_out_norm), axis=1, keepdims=True)
    out_with_tempreture = tf.keras.activations.sigmoid(user_pos_inner_product / temperature)
    pos_out = Reshape((1,))(out_with_tempreture)

    # user x neg_house inner product
    user_neg1_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_neg_1_out_norm), axis=1, keepdims=True)
    neg1_out = Reshape((1,))(tf.keras.activations.sigmoid(user_neg1_inner_product / temperature))

    user_neg2_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_neg_2_out_norm), axis=1, keepdims=True)
    neg2_out = Reshape((1,))(tf.keras.activations.sigmoid(user_neg2_inner_product / temperature))

    user_neg3_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_neg_3_out_norm), axis=1, keepdims=True)
    neg3_out = Reshape((1,))(tf.keras.activations.sigmoid(user_neg3_inner_product / temperature))

    user_neg4_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_neg_4_out_norm), axis=1, keepdims=True)
    neg4_out = Reshape((1,))(tf.keras.activations.sigmoid(user_neg4_inner_product / temperature))

    user_neg5_inner_product = tf.reduce_sum(tf.multiply(user_tower_out_norm, house_neg_5_out_norm), axis=1, keepdims=True)
    neg5_out = Reshape((1,))(tf.keras.activations.sigmoid(user_neg5_inner_product / temperature))

    # softmax 取user embedding与pos embedding计算的相似度做为期望预测的label概率
    softmax_inputs = {
        'pos': pos_out, 'neg1': neg1_out, 'neg2': neg2_out, 'neg3': neg3_out, 'neg4': neg4_out, 'neg5': neg5_out
    }
    out_logit = MySoftmax()(softmax_inputs)

    user_input = feature_columns['user_inputs']
    item_input = feature_columns['pos_house_inputs']

    user_output = user_tower_out_norm
    item_output = house_pos_out_norm

    user_model = Model(inputs=user_input, outputs=user_output)
    item_model = Model(inputs=item_input, outputs=item_output)
    all_model = Model(inputs=feature_columns['total_inputs'], outputs=out_logit)

    all_model.compile(
        loss=softmaxloss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[myLtrAcc],
        # run_eagerly=True
    )

    all_model.summary()

    return all_model, user_model, item_model
  
  
def myLtrAcc(y_true, y_pred):
    # pos的概率为最大则满足预期
    pred_max_index = tf.equal(tf.argmax(y_pred, axis=-1), 0) 
    correct_count = tf.reduce_sum(tf.cast(pred_max_index, tf.float32))
    return correct_count / tf.cast(len(pred_max_index), 'float32')
  
  
def softmaxloss(y_true, y_pred):
    pos_pred = tf.cast(tf.slice(y_pred, [0, 0], [-1, 1]), 'float32')
    return K.mean(-tf.math.log(pos_pred))  
  
  

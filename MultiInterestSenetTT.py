#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-03-02

"""
多兴趣SENet双塔模型Multi-Interest-Senet-Two-Towers (MISTT)

基于SENet，把User侧和Item侧的Embedding，打成多兴趣的。就是说，比如在用户侧塔，可以配置不同的SENet模块及对应的DNN结构，来强化不同方面兴趣的
Embedding表达。Item侧也可以如此办理，或者Item侧如果信息比较单一，可以仍然只打出一个Item Embedding，只需要维度上能和User侧多兴趣Embedding对齐即可
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from layers.tool_layers import *
from layers.model_layers import Tower, SENetLayer


# build model
def buildMISTT(temperature, city_dict, shangquan_dict, comm_dict, tower_num_layer, tower_num_layer_units):
    # define input
   ...


    user_feature_judge_1 = SENetLayer(last_shape=int(user_emb_feature_pooling.shape[-1]), reduction=16,
                                      name='user_embedding_senet_1')(user_emb_feature_pooling)

    user_feature_judge_2 = SENetLayer(last_shape=int(user_emb_feature_pooling.shape[-1]), reduction=16,
                                      name='user_embedding_senet_2')(user_emb_feature_pooling)

    user_embedding_senet_1 = []
    user_embedding_senet_2 = []
    for i in range(user_feature_judge_1.shape[1]):  # 2个senet特征权重维度一样，取其中一个即可
        x_1 = tf.slice(user_feature_judge_1, [0, i], [-1, 1])  # 取出senet1激活tensor中的第i个激活值
        x_2 = tf.slice(user_feature_judge_2, [0, i], [-1, 1])
        emb_with_jugde_1 = tf.multiply(user_emb_feature_list[i], x_1)
        emb_with_jugde_2 = tf.multiply(user_emb_feature_list[i], x_2)
        user_embedding_senet_1.append(emb_with_jugde_1)
        user_embedding_senet_2.append(emb_with_jugde_2)

    user_feature_1 = concatenate(user_embedding_senet_1, axis=1, name='user_feature_1')
    user_feature_2 = concatenate(user_embedding_senet_2, axis=1, name='user_feature_2')

    # 计算user tower的2个输出
    user_tower_out_1 = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                             activation=tf.nn.leaky_relu, name='user_tower_1')(user_feature_1)

    user_tower_out_2 = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                             activation=tf.nn.leaky_relu, name='user_tower_2')(user_feature_2)

    user_tower_out = Add(name="user_tower")([user_tower_out_1, user_tower_out_2])

    # item feature
    ...

    item_embed_features_judge_1 = SENetLayer(last_shape=int(item_embed_features_pooling.shape[-1]), reduction=16,
                                             name='item_embedding_senet_1')(item_embed_features_pooling)

    item_embed_features_judge_2 = SENetLayer(last_shape=int(item_embed_features_pooling.shape[-1]), reduction=16,
                                             name='item_embedding_senet_2')(item_embed_features_pooling)

    item_embedding_senet_1 = []
    item_embedding_senet_2 = []
    for i in range(item_embed_features_judge_1.shape[1]):
        x_1 = tf.slice(item_embed_features_judge_1, [0, i], [-1, 1])  # 取出senet激活tensor中的第i个激活值
        x_2 = tf.slice(item_embed_features_judge_2, [0, i], [-1, 1])
        emb_with_jugde_1 = tf.multiply(item_embedding_features_list[i], x_1)
        emb_with_jugde_2 = tf.multiply(item_embedding_features_list[i], x_2)
        item_embedding_senet_1.append(emb_with_jugde_1)
        item_embedding_senet_2.append(emb_with_jugde_2)

    item_feature_1 = concatenate(item_embedding_senet_1 + [item_prop_age, item_age_power2], axis=1,
                                 name='item_feature_1')
    item_feature_2 = concatenate(item_embedding_senet_2 + [item_prop_age, item_age_power2], axis=1,
                                 name='item_feature_2')

    # 计算2个item tower的输出
    item_tower_out_1 = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                             activation=tf.nn.leaky_relu, name='item_tower_1')(item_feature_1)
    item_tower_out_2 = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                             activation=tf.nn.leaky_relu, name='item_tower_2')(item_feature_2)

    item_tower_out = Add(name="item_tower")([item_tower_out_1, item_tower_out_2])

    # 计算内积，添加温度系数
    inner_product = tf.reduce_sum(tf.multiply(user_tower_out, item_tower_out), axis=1, keepdims=True)
    out_with_tempreture = tf.keras.activations.sigmoid(inner_product / temperature)
    out = Reshape((1,))(out_with_tempreture)

    user_input = [...]
    item_input = [...]

    user_output = user_tower_out
    item_output = item_tower_out

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

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
    user_city_seq = tf.keras.Input(shape=(5,), name='user_city_seq', dtype=tf.int64)
    user_shangquan_seq = tf.keras.Input(shape=(5,), name='user_shangquan_seq', dtype=tf.int64)
    user_comm_seq = tf.keras.Input(shape=(5,), name='user_comm_seq', dtype=tf.int64)
    user_price_seq = tf.keras.Input(shape=(5,), name='user_price_seq', dtype=tf.int64)
    user_area_seq = tf.keras.Input(shape=(5,), name='user_area_seq', dtype=tf.int64)

    item_city_id = tf.keras.Input(shape=(1,), name='city_id', dtype=tf.int64)
    item_comm_id = tf.keras.Input(shape=(1,), name='comm_id', dtype=tf.int64)
    item_shangquan_id = tf.keras.Input(shape=(1,), name='shangquan_id', dtype=tf.int64)
    item_price_id = tf.keras.Input(shape=(1,), name='price_id', dtype=tf.int64)
    item_area_id = tf.keras.Input(shape=(1,), name='area_id', dtype=tf.int64)
    item_floor_loc = tf.keras.Input(shape=(1,), name='floor_loc', dtype=tf.int64)
    item_total_floor = tf.keras.Input(shape=(1,), name='total_floor', dtype=tf.int64)
    item_room_num = tf.keras.Input(shape=(1,), name='room_num', dtype=tf.int64)
    item_hall = tf.keras.Input(shape=(1,), name='hall', dtype=tf.int64)
    item_bathroom = tf.keras.Input(shape=(1,), name='bathroom', dtype=tf.int64)
    item_prop_age = tf.keras.Input(shape=(1,), name='prop_age', dtype=tf.float32)

    # common emb  （目前只有一个城市的数据，适当调整dim)
    city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="city_emb")
    shangquan_Embedding = Embedding(input_dim=15000, output_dim=16, mask_zero=False, name="shangquan_emb")
    comm_Embedding = Embedding(input_dim=400000, output_dim=16, mask_zero=False, name="comm_emb")
    price_Embedding = Embedding(input_dim=50, output_dim=16, mask_zero=False, name="price_emb")
    area_Embedding = Embedding(input_dim=50, output_dim=16, mask_zero=False, name="area_emb")

    # user feature
    user_city_id_token = VocabLayer(city_dict, 'city_token')(user_city_seq)
    user_city_emb = city_Embedding(user_city_id_token)  # 以city_id为index取emb  shape(None, 5, emb_size)
    user_city_emb = GlobalAveragePooling1D()(user_city_emb)  # shape(None, emb_size)

    user_shangquan_id_token = VocabLayer(shangquan_dict, 'shangquan_token')(user_shangquan_seq)
    user_shangquan_emb = shangquan_Embedding(user_shangquan_id_token)
    user_shangquan_emb = GlobalAveragePooling1D()(user_shangquan_emb)

    user_comm_id_token = VocabLayer(comm_dict, 'comm_token')(user_comm_seq)
    user_comm_emb = comm_Embedding(user_comm_id_token)
    user_comm_emb = GlobalAveragePooling1D()(user_comm_emb)

    user_price_id_token = VocabLayer(price_dict, 'user_price_id_token')(user_price_seq)
    user_price_emb = price_Embedding(user_price_id_token)
    user_price_emb = GlobalAveragePooling1D()(user_price_emb)

    user_area_id_token = VocabLayer(area_dict, 'user_area_id_token')(user_area_seq)
    user_area_emb = area_Embedding(user_area_id_token)
    user_area_emb = GlobalAveragePooling1D()(user_area_emb)

    # item features
    item_city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="item_city_emb")
    item_shangquan_Embedding = Embedding(input_dim=15000, output_dim=16, mask_zero=False, name="item_shangquan_emb")
    item_comm_Embedding = Embedding(input_dim=400000, output_dim=16, mask_zero=False, name="item_comm_emb")

    item_city_id_token = VocabLayer(city_dict, 'city_token')(item_city_id)
    item_city_emb = item_city_Embedding(item_city_id_token)
    item_city_emb = Reshape((16,))(item_city_emb)

    item_shangquan_id_token = VocabLayer(shangquan_dict, 'shangquan_token')(item_shangquan_id)
    item_shangquan_emb = item_shangquan_Embedding(item_shangquan_id_token)
    item_shangquan_emb = Reshape((16,))(item_shangquan_emb)

    item_comm_id_token = VocabLayer(comm_dict, 'comm_token')(item_comm_id)
    item_comm_emb = item_comm_Embedding(item_comm_id_token)
    item_comm_emb = Reshape((16,))(item_comm_emb)

    item_price_emb = HashBucketsEmbedding(50, 16, name='item_price_emb')(item_price_id)
    item_price_emb = Reshape((16,))(item_price_emb)

    item_area_emb = HashBucketsEmbedding(50, 16, name='item_area_emb')(item_area_id)
    item_area_emb = Reshape((16,))(item_area_emb)

    item_floor_emb = HashBucketsEmbedding(50, 4, name='item_floor_emb')(item_floor_loc)
    item_floor_emb = Reshape((4,))(item_floor_emb)

    item_total_floor_emb = HashBucketsEmbedding(50, 4, name='item_total_floor_emb')(item_total_floor)
    item_total_floor_emb = Reshape((4,))(item_total_floor_emb)

    item_room_emb = HashBucketsEmbedding(10, 4, name='item_room_emb')(item_room_num)
    item_room_emb = Reshape((4,))(item_room_emb)

    item_hall_emb = HashBucketsEmbedding(10, 4, name='item_hall_emb')(item_hall)
    item_hall_emb = Reshape((4,))(item_hall_emb)

    item_bathroom_emb = HashBucketsEmbedding(10, 4, name='item_bathroom_emb')(item_bathroom)
    item_bathroom_emb = Reshape((4,))(item_bathroom_emb)

    # **********************************  表示层 **********************************#
    user_mlp_inputs = concatenate([user_city_emb, user_shangquan_emb, user_comm_emb, user_price_emb, user_area_emb],
                                  axis=-1, name='user_mlp_inputs')
    user_fm_inputs = tf.stack([user_city_emb, user_shangquan_emb, user_comm_emb, user_price_emb, user_area_emb],
                              axis=1, name='user_fm_inputs')
    user_dcn_inputs = concatenate([user_price_emb, user_area_emb], axis=-1, name="user_dcn_inputs")
    user_cin_inputs = tf.stack([user_city_emb, user_shangquan_emb, user_comm_emb], axis=1, name="user_cin_inputs")

    # 计算user_tower的并联输出
    user_mlp_dcn_out, user_fm_out, user_cin_out = parallel_layer(tower_num_layer, tower_num_layer_units,
                                                                 user_mlp_inputs, user_fm_inputs, user_dcn_inputs,
                                                                 user_cin_inputs)

    item_mlp_inputs = concatenate([item_city_emb, item_shangquan_emb, item_comm_emb, item_price_emb, item_area_emb,
                                   item_floor_emb, item_total_floor_emb, item_room_emb, item_hall_emb,
                                   item_bathroom_emb],
                                  axis=-1, name="item_mlp_inputs")
    item_fm_inputs = tf.stack([item_city_emb, item_shangquan_emb, item_comm_emb, item_price_emb, item_area_emb], axis=1,
                              name="item_fm_inputs")

    item_dcn_inputs = concatenate([item_price_emb, item_area_emb], axis=-1, name="item_dcn_inputs")
    item_cin_inputs = tf.stack([item_city_emb, item_shangquan_emb, item_comm_emb], axis=1, name="item_cin_inputs")

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

    user_input = [user_city_seq, user_shangquan_seq, user_comm_seq, user_price_seq, user_area_seq]
    item_input = [item_city_id, item_comm_id, item_shangquan_id, item_price_id, item_area_id, item_floor_loc,
                  item_total_floor, item_room_num, item_hall, item_bathroom, item_prop_age]

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

#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

"""
双塔模型 加入SENet
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from layers.tool_layers import *
from layers.model_layers import Tower, SENetLayer


# build model
def two_towers(temperature, city_dict, shangquan_dict, comm_dict, tower_num_layer, tower_num_layer_units):
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

    # common emb 区域类特征在底层做交互 （目前只有一个城市的数据，适当调整dim)
    city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="city_emb")
    shangquan_Embedding = Embedding(input_dim=15000, output_dim=32, mask_zero=False, name="shangquan_emb")
    comm_Embedding = Embedding(input_dim=400000, output_dim=32, mask_zero=False, name="comm_emb")

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

    # user bilstm fea
    user_shangquan_seq_emb = Bidirectional(LSTM(32, return_sequences=False),
                                           merge_mode='concat', name='bilstm_shangquan_fea')(
        Reshape((-1, 16))(user_shangquan_emb))
    # user_shangquan_seq_emb = Dropout(0.2)(user_shangquan_seq_emb, training=True)
    user_shangquan_seq_emb = Reshape((64,))(user_shangquan_seq_emb)

    user_comm_seq_emb = Bidirectional(LSTM(32, return_sequences=False),
                                      merge_mode='concat', name='bilstm_comm_fea')(Reshape((-1, 16))(user_comm_emb))
    # user_comm_seq_emb = Dropout(0.2)(user_comm_seq_emb, training=True)
    user_comm_seq_emb = Reshape((64,))(user_comm_seq_emb)

    # concat user features
    user_emb_feature_list = [user_city_emb, user_shangquan_emb, user_comm_emb,
                             user_shangquan_seq_emb, user_comm_seq_emb]
    user_emb_feature_pooling = [Reshape((1,))(tf.reduce_mean(i, 1)) for i in user_emb_feature_list]  # 1 * feature_num
    user_emb_feature_pooling = concatenate(user_emb_feature_pooling, axis=1)

    user_feature_judge = SENetLayer(last_shape=int(user_emb_feature_pooling.shape[-1]), reduction=16,
                                    name='user_embedding_senet')(user_emb_feature_pooling)

    user_embedding_senet = []
    for i in range(user_feature_judge.shape[1]):
        x = tf.slice(user_feature_judge, [0, i], [-1, 1])  # 取出senet激活tensor中的第i个激活值
        emb_with_jugde = tf.multiply(user_emb_feature_list[i], x)
        user_embedding_senet.append(emb_with_jugde)

    user_feature = concatenate(user_embedding_senet, axis=1, name='user_feature')

    # 计算user tower的输出
    user_tower_out = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                           activation=tf.nn.leaky_relu, name='user_tower')(user_feature)
    #     user_tower_out_l2 = Dense(32, activation='relu', name="user_tower", activity_regularizer='l2')(user_tower_out)

    # item feature
    item_city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=False, name="item_city_emb")
    item_shangquan_Embedding = Embedding(input_dim=15000, output_dim=32, mask_zero=False, name="item_shangquan_emb")
    item_comm_Embedding = Embedding(input_dim=400000, output_dim=32, mask_zero=False, name="item_comm_emb")

    item_city_id_token = VocabLayer(city_dict, 'city_token')(item_city_id)
    item_city_emb = item_city_Embedding(item_city_id_token)
    item_city_emb = Reshape((16,))(item_city_emb)

    item_shangquan_id_token = VocabLayer(shangquan_dict, 'shangquan_token')(item_shangquan_id)
    item_shangquan_emb = item_shangquan_Embedding(item_shangquan_id_token)
    item_shangquan_emb = Reshape((32,))(item_shangquan_emb)

    item_comm_id_token = VocabLayer(comm_dict, 'comm_token')(item_comm_id)
    item_comm_emb = item_comm_Embedding(item_comm_id_token)
    item_comm_emb = Reshape((32,))(item_comm_emb)

    item_price_emb = HashBucketsEmbedding(50, 4, name='item_price_emb')(item_price_id)
    item_price_emb = Reshape((4,))(item_price_emb)

    item_area_emb = HashBucketsEmbedding(50, 4, name='item_area_emb')(item_area_id)
    item_area_emb = Reshape((4,))(item_area_emb)

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

    # item age
    item_age_power2 = Power_layer(2)(item_prop_age)

    item_embedding_features_list = [item_city_emb, item_shangquan_emb, item_comm_emb, item_price_emb,
                                    item_area_emb, item_floor_emb, item_total_floor_emb, item_room_emb,
                                    item_hall_emb, item_bathroom_emb]

    item_embed_features_pooling = [Reshape((1,))(tf.reduce_mean(i, 1)) for i in
                                   item_embedding_features_list]  # 1 * feature_num
    item_embed_features_pooling = concatenate(item_embed_features_pooling, axis=1)

    item_embed_features_judge = SENetLayer(last_shape=int(item_embed_features_pooling.shape[-1]), reduction=16,
                                           name='item_embedding_senet')(item_embed_features_pooling)

    item_embedding_senet = []
    for i in range(item_embed_features_judge.shape[1]):
        x = tf.slice(item_embed_features_judge, [0, i], [-1, 1])  # 取出senet激活tensor中的第i个激活值
        emb_with_jugde = tf.multiply(item_embedding_features_list[i], x)
        item_embedding_senet.append(emb_with_jugde)

    item_feature = concatenate(item_embedding_senet + [item_prop_age, item_age_power2], axis=1,
                               name='item_feature')

    # 计算item tower的输出
    item_tower_out = Tower(layer_num=tower_num_layer, layer_units=tower_num_layer_units,
                           activation=tf.nn.leaky_relu, name='item_tower')(item_feature)
    #     item_tower_out_l2 = Dense(32, activation='relu', name="item_tower", activity_regularizer='l2')(item_tower_out)

    # 计算内积，添加温度系数
    inner_product = tf.reduce_sum(tf.multiply(user_tower_out, item_tower_out), axis=1, keepdims=True)
    out_with_tempreture = tf.keras.activations.sigmoid(inner_product / temperature)
    out = Reshape((1,))(out_with_tempreture)

    user_input = [user_city_seq, user_shangquan_seq, user_comm_seq, user_price_seq, user_area_seq]
    item_input = [item_city_id, item_comm_id, item_shangquan_id, item_price_id, item_area_id, item_floor_loc,
                  item_total_floor, item_room_num, item_hall, item_bathroom, item_prop_age]

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
#!/usr/bin/env python
# coding: utf-8
# author: stefan 2023-02-21


from layers.tool_layers import *
from layers.model_layers import GlobalAveragePooling1DSef


def build_feature_column(city_dict, shangquan_dict, comm_dict, price_dict, area_dict):
    # define input
    # user
    user_city_seq = tf.keras.Input(shape=(5,), name='user_city_seq', dtype=tf.int64)
    user_shangquan_seq = tf.keras.Input(shape=(5,), name='user_shangquan_seq', dtype=tf.int64)
    user_comm_seq = tf.keras.Input(shape=(5,), name='user_comm_seq', dtype=tf.int64)
    user_price_seq = tf.keras.Input(shape=(5,), name='user_price_seq', dtype=tf.int64)
    user_area_seq = tf.keras.Input(shape=(5,), name='user_area_seq', dtype=tf.int64)

    # house pos ft
    city_id = tf.keras.Input(shape=(1,), name='city_id', dtype=tf.int64)
    comm_id = tf.keras.Input(shape=(1,), name='comm_id', dtype=tf.int64)


    # house neg n1
    city_id_n1 = tf.keras.Input(shape=(1,), name='city_id_n1', dtype=tf.int64)
    comm_id_n1 = tf.keras.Input(shape=(1,), name='comm_id_n1', dtype=tf.int64)


    # house neg n2
    city_id_n2 = tf.keras.Input(shape=(1,), name='city_id_n2', dtype=tf.int64)
    comm_id_n2 = tf.keras.Input(shape=(1,), name='comm_id_n2', dtype=tf.int64)


    # house neg n3
    city_id_n3 = tf.keras.Input(shape=(1,), name='city_id_n3', dtype=tf.int64)
    comm_id_n3 = tf.keras.Input(shape=(1,), name='comm_id_n3', dtype=tf.int64)


    # house neg n4
    city_id_n4 = tf.keras.Input(shape=(1,), name='city_id_n4', dtype=tf.int64)
    comm_id_n4 = tf.keras.Input(shape=(1,), name='comm_id_n4', dtype=tf.int64)


    # house neg n5
    city_id_n5 = tf.keras.Input(shape=(1,), name='city_id_n5', dtype=tf.int64)
    comm_id_n5 = tf.keras.Input(shape=(1,), name='comm_id_n5', dtype=tf.int64)


    # common emb 区域类特征在底层做交互
    city_Embedding = Embedding(input_dim=400, output_dim=16, mask_zero=True, name="city_emb")
    comm_Embedding = Embedding(input_dim=400000, output_dim=32, mask_zero=True, name="comm_emb")


    # user feature
    user_city_id_token = VocabLayer(city_dict, 'city_token')(user_city_seq)
    user_city_emb_seq = city_Embedding(user_city_id_token)  # 以city_id为index取emb  shape(None, 5, emb_size)
    user_city_emb = GlobalAveragePooling1DSef()(user_city_emb_seq)  # shape(None, emb_size)


    user_comm_id_token = VocabLayer(comm_dict, 'comm_token')(user_comm_seq)
    user_comm_emb_seq = comm_Embedding(user_comm_id_token)
    user_comm_emb = GlobalAveragePooling1DSef()(user_comm_emb_seq)



    # concat user features
    user_feature = concatenate([user_city_emb
                                   # , user_shangquan_emb, user_comm_emb, user_price_emb, user_area_emb
                                ], axis=1,
                               name='user_feature')

    # house pos features
    pos_city_id_token = VocabLayer(city_dict, 'pos_city_token')(city_id)
    pos_city_emb = item_city_Embedding(pos_city_id_token)
    pos_city_emb = Reshape((16,))(pos_city_emb)


    pos_comm_id_token = VocabLayer(comm_dict, 'pos_comm_token')(comm_id)
    pos_comm_emb = item_comm_Embedding(pos_comm_id_token)
    pos_comm_emb = Reshape((32,))(pos_comm_emb)



    pos_item_feaure = concatenate([pos_city_emb
                                   #    , pos_shangquan_emb, pos_comm_emb, pos_price_emb, pos_area_emb,
                                   # item_floor_emb, item_room_emb, item_hall_emb, item_bathroom_emb, pqs
                                   ], axis=1,
                                  name='pos_item_feaure')

    # house neg1 features
    neg_city_id_token1 = VocabLayer(city_dict, 'neg_city_id_token1')(city_id_n1)
    neg_city_emb1 = item_city_Embedding(neg_city_id_token1)
    neg_city_emb1 = Reshape((16,))(neg_city_emb1)


    neg_comm_id_token1 = VocabLayer(comm_dict, 'neg_comm_id_token1')(comm_id_n1)
    neg_comm_emb1 = item_comm_Embedding(neg_comm_id_token1)
    neg_comm_emb1 = Reshape((32,))(neg_comm_emb1)


    neg_item_feature_1 = concatenate([neg_city_emb1
                                      #    , neg_shangquan_emb1, neg_comm_emb1, neg_price_emb1, neg_area_emb1,
                                      # item_floor_emb1, item_room_emb1, item_hall_emb1, item_bathroom_emb1, pqs_n1
                                      ], axis=1,
                                  name='neg_item_feature_1')

    # house neg2 features
    neg_city_id_token2 = VocabLayer(city_dict, 'neg_city_id_token2')(city_id_n2)
    neg_city_emb2 = item_city_Embedding(neg_city_id_token2)
    neg_city_emb2 = Reshape((16,))(neg_city_emb2)


    neg_comm_id_token2 = VocabLayer(comm_dict, 'neg_comm_id_token2')(comm_id_n2)
    neg_comm_emb2 = item_comm_Embedding(neg_comm_id_token2)
    neg_comm_emb2 = Reshape((32,))(neg_comm_emb2)



    neg_item_feature_2 = concatenate([neg_city_emb2, neg_shangquan_emb2, neg_comm_emb2, neg_price_emb2, neg_area_emb2,
                                      item_floor_emb2, item_room_emb2, item_hall_emb2, item_bathroom_emb2, pqs_n2],
                                     axis=1,
                                     name='neg_item_feature_2')

    # house neg3 features
    neg_city_id_token3 = VocabLayer(city_dict, 'neg_city_id_token3')(city_id_n3)
    neg_city_emb3 = item_city_Embedding(neg_city_id_token3)
    neg_city_emb3 = Reshape((16,))(neg_city_emb3)



    neg_comm_id_token3 = VocabLayer(comm_dict, 'neg_comm_id_token3')(comm_id_n3)
    neg_comm_emb3 = item_comm_Embedding(neg_comm_id_token3)
    neg_comm_emb3 = Reshape((32,))(neg_comm_emb3)


    neg_item_feature_3 = concatenate([neg_city_emb3, neg_shangquan_emb3, neg_comm_emb3, neg_price_emb3, neg_area_emb3,
                                      item_floor_emb3, item_room_emb3, item_hall_emb3, item_bathroom_emb3, pqs_n3],
                                     axis=1,
                                     name='neg_item_feature_3')

    # house neg4 features
    neg_city_id_token4 = VocabLayer(city_dict, 'neg_city_id_token4')(city_id_n4)
    neg_city_emb4 = item_city_Embedding(neg_city_id_token4)
    neg_city_emb4 = Reshape((16,))(neg_city_emb4)



    neg_comm_id_token4 = VocabLayer(comm_dict, 'neg_comm_id_token4')(comm_id_n4)
    neg_comm_emb4 = item_comm_Embedding(neg_comm_id_token4)
    neg_comm_emb4 = Reshape((32,))(neg_comm_emb4)



    neg_item_feature_4 = concatenate([neg_city_emb4
                                      #    , neg_shangquan_emb4, neg_comm_emb4, neg_price_emb4, neg_area_emb4,
                                      # item_floor_emb4, item_room_emb4, item_hall_emb4, item_bathroom_emb4, pqs_n4
                                      ],
                                     axis=1,
                                     name='neg_item_feature_4')

    # house neg5 features
    neg_city_id_token5 = VocabLayer(city_dict, 'neg_city_id_token5')(city_id_n5)
    neg_city_emb5 = item_city_Embedding(neg_city_id_token5)
    neg_city_emb5 = Reshape((16,))(neg_city_emb5)



    neg_comm_id_token5 = VocabLayer(comm_dict, 'neg_comm_id_token5')(comm_id_n5)
    neg_comm_emb5 = item_comm_Embedding(neg_comm_id_token5)
    neg_comm_emb5 = Reshape((32,))(neg_comm_emb5)


    neg_item_feature_5 = concatenate([neg_city_emb5, neg_shangquan_emb5, neg_comm_emb5, neg_price_emb5, neg_area_emb5,
                                      item_floor_emb5, item_room_emb5, item_hall_emb5, item_bathroom_emb5, pqs_n5],
                                     axis=1,
                                     name='neg_item_feature_5')

    user_inputs = [user_city_seq, user_shangquan_seq, user_comm_seq, user_price_seq, user_area_seq]
    pos_house_inputs = [city_id, comm_id, shangquan_id, price_id, area_id, floor_loc, room_num, hall, bathroom, pqs]
    neg_house_1_inputs = [city_id_n1, comm_id_n1, shangquan_id_n1, price_id_n1, area_id_n1, floor_loc_n1, room_num_n1,
                          hall_n1, bathroom_n1, pqs_n1]
    neg_house_2_inputs = [city_id_n2, comm_id_n2, shangquan_id_n2, price_id_n2, area_id_n2, floor_loc_n2, room_num_n2,
                          hall_n2, bathroom_n2, pqs_n2]
    neg_house_3_inputs = [city_id_n3, comm_id_n3, shangquan_id_n3, price_id_n3, area_id_n3, floor_loc_n3, room_num_n3,
                          hall_n3, bathroom_n3, pqs_n3]
    neg_house_4_inputs = [city_id_n4, comm_id_n4, shangquan_id_n4, price_id_n4, area_id_n4, floor_loc_n4, room_num_n4,
                          hall_n4, bathroom_n4, pqs_n4]
    neg_house_5_inputs = [city_id_n5, comm_id_n5, shangquan_id_n5, price_id_n5, area_id_n5, floor_loc_n5, room_num_n5,
                          hall_n5, bathroom_n5, pqs_n5]

    result = {
        'user_inputs':user_inputs,
        'pos_house_inputs':pos_house_inputs,
        'neg_house_1_inputs':neg_house_1_inputs,
        'neg_house_2_inputs':neg_house_2_inputs,
        'neg_house_3_inputs':neg_house_3_inputs,
        'neg_house_4_inputs':neg_house_4_inputs,
        'neg_house_5_inputs':neg_house_5_inputs,
        'total_inputs':user_inputs + pos_house_inputs + neg_house_1_inputs + neg_house_2_inputs + neg_house_3_inputs +
                       neg_house_4_inputs + neg_house_5_inputs,
        'user_ft':user_feature,
        'pos_house_ft':pos_item_feaure,
        'neg_house_1_ft':neg_item_feature_1,
        'neg_house_2_ft': neg_item_feature_2,
        'neg_house_3_ft': neg_item_feature_3,
        'neg_house_4_ft': neg_item_feature_4,
        'neg_house_5_ft': neg_item_feature_5
    }

    return result


#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

import tensorflow as tf

rank_909_config = {
    # user
    "user_city_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_shangquan_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_comm_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_price_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_area_seq": tf.io.FixedLenFeature([5], tf.int64),

    # prop
    "city_id": tf.io.FixedLenFeature([1], tf.int64),
    "comm_id": tf.io.FixedLenFeature([1], tf.int64),
    "shangquan_id": tf.io.FixedLenFeature([1], tf.int64),
    "price_id": tf.io.FixedLenFeature([1], tf.int64),
    "area_id": tf.io.FixedLenFeature([1], tf.int64),
    "floor_loc": tf.io.FixedLenFeature([1], tf.int64),
    "total_floor": tf.io.FixedLenFeature([1], tf.int64),
    "room_num": tf.io.FixedLenFeature([1], tf.int64),
    "hall": tf.io.FixedLenFeature([1], tf.int64),
    "bathroom": tf.io.FixedLenFeature([1], tf.int64),

    "pqs": tf.io.FixedLenFeature([1], tf.float32),
    "prop_age": tf.io.FixedLenFeature([1], tf.float32),
    "edu_link_rate": tf.io.FixedLenFeature([1], tf.float32),
    "floor_link_rate": tf.io.FixedLenFeature([1], tf.float32),

    "orient": tf.io.FixedLenFeature([1], tf.int64),
    "fitment": tf.io.FixedLenFeature([1], tf.int64),
    "is_guarantee": tf.io.FixedLenFeature([1], tf.int64),
    "is_media": tf.io.FixedLenFeature([1], tf.int64),
    "is_720": tf.io.FixedLenFeature([1], tf.int64),

    "green_rate": tf.io.FixedLenFeature([1], tf.float32),
    "traffic": tf.io.FixedLenFeature([1], tf.float32),
    "education": tf.io.FixedLenFeature([1], tf.float32),
    "business": tf.io.FixedLenFeature([1], tf.float32),
    "environment": tf.io.FixedLenFeature([1], tf.float32),
    "popularity": tf.io.FixedLenFeature([1], tf.float32),
    "impression_score": tf.io.FixedLenFeature([1], tf.float32),
    "comm_score": tf.io.FixedLenFeature([1], tf.float32),

    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64),

}

recall_909_config = {
    # user
    "user_city_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_shangquan_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_comm_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_price_seq": tf.io.FixedLenFeature([5], tf.int64),
    "user_area_seq": tf.io.FixedLenFeature([5], tf.int64),

    # prop
    "city_id": tf.io.FixedLenFeature([1], tf.int64),
    "comm_id": tf.io.FixedLenFeature([1], tf.int64),
    "shangquan_id": tf.io.FixedLenFeature([1], tf.int64),
    "price_id": tf.io.FixedLenFeature([1], tf.int64),
    "area_id": tf.io.FixedLenFeature([1], tf.int64),
    "floor_loc": tf.io.FixedLenFeature([1], tf.int64),
    "total_floor": tf.io.FixedLenFeature([1], tf.int64),
    "room_num": tf.io.FixedLenFeature([1], tf.int64),
    "hall": tf.io.FixedLenFeature([1], tf.int64),
    "bathroom": tf.io.FixedLenFeature([1], tf.int64),
    "prop_age": tf.io.FixedLenFeature([1], tf.float32),

    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64),

}

data_config = {
    "909-recall": recall_909_config,
    "909-rank": rank_909_config
}

#!/usr/bin/env python
# coding: utf-8

# import sys
from data_process.common_utils import read_data_ltr
from data_process.data_config import data_config
from recall.LTR import config
from recall.LTR.PairWiseTower import build_ltr_tower
import tensorflow as tf

FLAGS = config.FLAGS

# read data
train_set = read_data_ltr(path=FLAGS.train_data, class_num=FLAGS.class_num, batch_size=FLAGS.batch_size, if_shuffle=True,
                          feat_desc=data_config['recall-ltr'])

test_set = read_data_ltr(path=FLAGS.test_data, class_num=FLAGS.class_num, batch_size=FLAGS.batch_size, if_shuffle=False,
                          feat_desc=data_config['recall-ltr'])

# define model
all_model, user_model, item_model = build_ltr_tower(FLAGS.temperature, FLAGS.city_dict, FLAGS.shangquan_dict, FLAGS.comm_dict, FLAGS.price_dict,
                        FLAGS.area_dict, FLAGS.tower_num_layer, FLAGS.tower_num_layer_units)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.online_logs, embeddings_freq=1,
                                                      embeddings_data=train_set)

all_model.fit(
    x=train_set,
    epochs=FLAGS.epoch,
    callbacks=[tensorboard_callback]
)

# save models
item_model.save(FLAGS.item_model_pb, save_format='tf')  # 保存item model的weights用于离线获取emb

user_model.save(FLAGS.user_model_pb, save_format='tf')  # 保存user model的pb模型用于在线预测

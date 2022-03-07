#!/usr/bin/env python
# coding: utf-8

# import sys
# sys.path.insert(0, r'/code/Stefan/909_recall/dm-recommend-tf2/') # 线上要加入搜索目录的路径

from data_process.common_utils import *
from data_process.data_config import *
from recall import config
from recall import TwoTowers, TwoTowersSENet, MultiInterestSenetTT, TwoResNet, ParallelTowers

FLAGS = config.FLAGS

# read data
train_set = read_data(path=FLAGS.train_data, batch_size=FLAGS.batch_size, if_shuffle=True,
                      feat_desc=data_config["909-recall"])
test_set = read_data(path=FLAGS.eval_data, batch_size=FLAGS.batch_size, feat_desc=data_config["909-recall"])

# define models
if FLAGS.if_use_senet:
    all_model, user_model, item_model = TwoTowersSENet.two_towers(FLAGS.temperature, FLAGS.city_dict,
                                                                  FLAGS.shangquan_dict,
                                                                  FLAGS.comm_dict, FLAGS.tower_num_layer,
                                                                  FLAGS.tower_num_layer_units.split(','))
elif FLAGS.if_use_MISTT:
    all_model, user_model, item_model = MultiInterestSenetTT.buildMISTT(FLAGS.temperature, FLAGS.city_dict,
                                                                        FLAGS.shangquan_dict,
                                                                        FLAGS.comm_dict, FLAGS.tower_num_layer,
                                                                        FLAGS.tower_num_layer_units.split(','))
elif FLAGS.if_use_twoResNet:
    all_model, user_model, item_model = TwoResNet.two_towers(FLAGS.temperature, FLAGS.city_dict,
                                                             FLAGS.shangquan_dict,
                                                             FLAGS.comm_dict, FLAGS.tower_num_layer,
                                                             FLAGS.tower_num_layer_units.split(','))
elif FLAGS.if_use_parallel:
    all_model, user_model, item_model = ParallelTowers.parallel_towers(FLAGS.temperature, FLAGS.city_dict,
                                                                       FLAGS.shangquan_dict, FLAGS.comm_dict,
                                                                       FLAGS.price_dict, FLAGS.area_dict,
                                                                       FLAGS.tower_num_layer,
                                                                       FLAGS.tower_num_layer_units.split(','))
else:
    all_model, user_model, item_model = TwoTowers.two_towers(FLAGS.temperature, FLAGS.city_dict,
                                                             FLAGS.shangquan_dict,
                                                             FLAGS.comm_dict, FLAGS.tower_num_layer,
                                                             FLAGS.tower_num_layer_units.split(','))

# define callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.online_logs, embeddings_freq=1,
                                                      embeddings_data=train_set)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=8)  # 早停法，防止过拟合
plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", verbose=1, mode='max', factor=0.5,
                                               patience=2)  # 当评价指标不在提升时，减少学习率

batch_print_callback = tf.keras.callbacks.LambdaCallback(
    on_batch_begin=lambda batch, logs: print(batch))

# run train
all_model.fit(
    x=train_set,
    epochs=FLAGS.epoch,
    validation_data=test_set,
    callbacks=[tensorboard_callback, early_stopping, plateau, batch_print_callback]
)

# save models
item_model.save_weights(FLAGS.item_model_weights)  # 保存item model的weights用于离线获取emb

item_model.save(FLAGS.item_model_pb, save_format='tf')  # 保存item model的weights用于离线获取emb

user_model.save(FLAGS.user_model_pb, save_format='tf')  # 保存user model的pb模型用于在线预测

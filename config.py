#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

import tensorflow as tf
import datetime

"""
recall
模型相关参数配置
"""

flags = tf.compat.v1.flags

flags.DEFINE_boolean("if_use_senet", False, "plus senet module.")
flags.DEFINE_boolean("if_use_MISTT", False, "plus MISTT module.")
flags.DEFINE_boolean("if_use_twoResNet", False, "plus ResNet module.")
flags.DEFINE_boolean("if_use_parallel", True, "plus parallel module.")

flags.DEFINE_string("item_model_pb", "./item_model_pb", "Base directory for the item model.")
flags.DEFINE_string("user_model_pb", "./user_model_pb", "Base directory for the user model.")
flags.DEFINE_string("item_model_weights", "./item_model_weights", "Base directory for the item model weights.")

flags.DEFINE_string("city_dict", "../demo_data/city_dict", "Path to the city_dict.")
flags.DEFINE_string("shangquan_dict", "../demo_data/shangquan_dict", "Path to the shangquan_dict.")
flags.DEFINE_string("comm_dict", "../demo_data/comm_dict", "Path to the comm_dict.")
flags.DEFINE_string("price_dict", "../demo_data/price_dict", "Path to the price_dict.")
flags.DEFINE_string("area_dict", "../demo_data/area_dict", "Path to the area_dict.")

flags.DEFINE_string("train_data", "../demo_data/part-r-00003-Copy1", "Path to the train data")
flags.DEFINE_string("eval_data", "../demo_data/part-r-00003-Copy1", "Path to the evaluation data.")

flags.DEFINE_string("online_logs", "../online_logs", "Path to the log.")

flags.DEFINE_integer("batch_size", 1024, "Training batch size")  # 40960
flags.DEFINE_integer("epoch", 1, "Training epochs")    # 40
flags.DEFINE_float("temperature", 0.001, "temperature")
flags.DEFINE_integer("tower_num_layer", 3, "num of layers")
flags.DEFINE_string("tower_num_layer_units", "256,128,64", "hidden units of layers")
flags.DEFINE_string("cin_size", "64,64", "a list of the number of layers")

FLAGS = flags.FLAGS

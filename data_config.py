#!/usr/bin/env python
# coding: utf-8
# author: stefan 2022-02-28

import tensorflow as tf

recall_config = {
    # user


    # prop


    # label
    "is_click": tf.io.FixedLenFeature([], tf.int64),

}


data_config = {
    "recall": recall_config,
}

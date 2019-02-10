#coding=utf-8
# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import time
import math
import random

import tensorflow as tf
import numpy

from model import *
from utils import *
from sample_io import SampleIO

import xdl
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook
import numpy as np

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def get_data_prefix():
    return xdl.get_config('data_dir')


train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")
test_file = os.path.join(get_data_prefix(), "local_test_splitByUser")
uid_voc = os.path.join(get_data_prefix(), "uid_voc.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')


def train(train_file=train_file,
          test_file=test_file,
          uid_voc=uid_voc,
          mid_voc=mid_voc,
          cat_voc=cat_voc,
          item_info=item_info,
          reviews_info=reviews_info,
          batch_size=128,
          maxlen=200,
          test_iter=700):
    if xdl.get_config('model') == 'din':
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'dien':
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien')

    sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,
                         cat_voc, item_info, reviews_info,
                         batch_size, maxlen, EMBEDDING_DIM)
    with xdl.model_scope('train'):
        train_ops = model.build_final_net(EMBEDDING_DIM, sample_io)
        lr = 0.001
        # Adam Adagrad
        train_ops.append(xdl.Adam(lr).optimize())
        hooks = []
        log_format = "[%(time)s] lstep[%(lstep)s] gstep[%(gstep)s] lqps[%(lqps)s] gqps[%(gqps)s] loss[%(loss)s]"
        hooks = [QpsMetricsHook(), MetricsPrinterHook(log_format)]

        if xdl.get_task_index() == 0:
            hooks.append(xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_interval')))
        train_sess = xdl.TrainSession(hooks=hooks)

    with xdl.model_scope('test'):
        test_ops = model.build_final_net(EMBEDDING_DIM, sample_io, is_train=False)
        test_sess = xdl.TrainSession()

    model.run(train_ops, train_sess, test_ops, test_sess, test_iter=test_iter)


# 可以跑起来的test模式
def test(train_file=train_file,
         test_file=test_file,
         uid_voc=uid_voc,
         mid_voc=mid_voc,
         cat_voc=cat_voc,
         batch_size=128,
         maxlen=100):
    # sample_io
    sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,
                         cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)

    if xdl.get_config('model') == 'din':
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'dien':
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien model')

    @xdl.tf_wrapper(is_training=False)
    def tf_test_model(*inputs):
        with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
            model.build_tf_net(inputs, False)
        test_ops = model.test_ops()
        return test_ops[0], test_ops[1:]

    # test
    datas = sample_io.next_test()
    test_ops = tf_test_model(
        *model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))

    #saver = xdl.Saver()
    # checkpoint_version ="./ckpt_dir/ckpt-................8700/" # ckpt_version
    #saver.restore(version="ckpt-................8700")  # version=
    eval_sess = xdl.TrainSession()

    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' %
          eval_model(eval_sess, test_ops))


def predict(train_file=test_file,
            test_file=test_file,
            uid_voc=uid_voc,
            mid_voc=mid_voc,
            cat_voc=cat_voc,
            item_info=item_info,
            reviews_info=reviews_info,
            batch_size=16,
            maxlen=100):
    # sample_io
    sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,
                         cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)

    if xdl.get_config('model') == 'din':
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'dien':
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien model')

    @xdl.tf_wrapper(is_training=False)
    def tf_test_model(*inputs):
        with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
            model.build_tf_net(inputs, False)
        predict_ops = model.predict_ops()
        return predict_ops[0], predict_ops[1:]

    # predict
    datas = sample_io.next_predict()
    predict_ops = tf_test_model(
        *model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))  # predict_ops中包含有uuid

    saver = xdl.Saver()
    saver.restore(version="ckpt-................3000")

    eval_sess = xdl.TrainSession()

    stored_arr = predict_model(eval_sess, predict_ops)
    cnt = 0
    fw=open("predict_result.txt",'a+')
    for r in stored_arr:
	fw.write("%s\t%s\t%s\t%s\t%s\n"%(str(r[0]),str(r[1]),str(r[2]),str(r[3]),str(r[4])))
        cnt += 1
        if cnt < 10:
            print(r[0], r[1], r[2], r[3])
    fw.close()


if __name__ == '__main__':
    SEED = xdl.get_config("seed")
    if SEED is None:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    job_type = xdl.get_config("job_type")
    if job_type == 'train':
        train()
    elif job_type == 'test':
        test()
    elif job_type == "predict":
        predict()
    else:
        print('job type must be train or test, do nothing...')

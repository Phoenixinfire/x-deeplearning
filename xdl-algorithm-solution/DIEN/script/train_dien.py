# coding=utf-8
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
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" 


import sys
import time
import math
import random
import datetime
import tensorflow as tf
import numpy
from multiprocessing import Pool
from model import *
from utils import *
from sample_io import SampleIO
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import xdl
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook
import numpy as np

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0



now_time=datetime.datetime.now()
yes_time=now_time+datetime.timedelta(days=-1)

#yes_d=yes_time.strftime("%Y-%m-%d")

yes_d=xdl.get_app_id()
print(yes_d)

def get_data_prefix():
    return os.path.join(xdl.get_config('data_dir'),"dt=%s"%(yes_d))

print(get_data_prefix())


train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")
test_file = os.path.join(get_data_prefix(), "local_test_splitByUser")
predict_file = os.path.join(get_data_prefix(), "predict")
predict_result_file = os.path.join(get_data_prefix(), "predict_result")
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
    ids, datas = sample_io.next_test()
    test_ops = tf_test_model(
        *model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))

    # saver = xdl.Saver()
    # checkpoint_version ="./ckpt_dir/ckpt-................8700/" # ckpt_version
    # saver.restore(version="ckpt-................8700")  # version=
    eval_sess = xdl.TrainSession()

    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' %
          eval_model(eval_sess, test_ops))


def predict(train_file=predict_file,
            test_file=predict_file,
            uid_voc=uid_voc,
            mid_voc=mid_voc,
            cat_voc=cat_voc,
            item_info=item_info,
            reviews_info=reviews_info,
            batch_size=128,
            maxlen=100,
            day="default"):
    # sample_io
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

    files = os.listdir(predict_file)
    for f in files:
        print(f)
        eval_sess = xdl.TrainSession()
        print("model")
        saver = xdl.Saver()
        saver.restore(version="ckpt-................5000")
        print("predict_start")
        if f != "_SUCCESS":
            abs_file = predict_file + "/%s" % (f)
            print(abs_file)
            sample_io = SampleIO(abs_file, abs_file, uid_voc, mid_voc,
                                 cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)
            # predict
            print("data_next")
            ids, datas = sample_io.next_predict()
            predict_ops = tf_test_model(
                *model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))  # predict_ops中包含有uuid
            print("predict_real_start")
            stored_arr = predict_model(eval_sess, predict_ops)
            cnt = 0
            print("predict_finish")
            fw = open("%s/predict_result_tag_%s_%s.txt" % (predict_result_file, day, f), 'a+')
            for r in stored_arr:
                fw.write("%s\t%s\n" % (str(r[0]), str(r[1])))
                cnt += 1
                if cnt < 10:
                    print(r[0], r[1], r[2], r[3])
            fw.close()

def predict_each_core(sess, file_to_predict, predict_result_file, day, model_str, uid_voc, mid_voc,
                      cat_voc, item_info, reviews_info, maxlen, index):
    
    try:
         # sample_io
        if xdl.get_config('model') == 'din':
            model = Model_DIN(
                EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif xdl.get_config('model') == 'dien':
            model = Model_DIEN(
                EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            raise Exception('only support din and dien model')

       
	logger.info("input")
	sample_io = SampleIO(file_to_predict, file_to_predict, uid_voc, mid_voc,
	                     cat_voc, item_info, reviews_info,1 , maxlen, EMBEDDING_DIM)
	idx_ops, datas = sample_io.next_predict()  
	logger.info("data_size=%d=%d"%(len(idx_ops),len(datas)))
	
	@xdl.tf_wrapper(is_training=False, gpu_memory_fraction=0.9, device_type="gpu")
	def tf_test_model(*inputs):
	    with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
	        model.build_tf_net(inputs, False)
	    predict_ops = model.predict_ops()
	    return predict_ops[0], predict_ops[1:]
	
	predict_ops = tf_test_model(*model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))  # predict_ops中包含有uuid
	logger.info("hahaha")
	nums = 0
	stored_arr = []
	logger.info("hello,xdl")
	while not sess.should_stop():
	    logger.info("wearhear")
	    nums += 1

	    values, ids = sess.run([predict_ops, idx_ops])
	    logger.info("runsucdess")
	    uid, mid, cat = ids  # sess.run(idx_ops)
	    if values is None:
	        break
	    prob, target = values
	    prob_1 = [x[0] for x in prob]
	    prob_0 = [x[1] for x in prob]
	    cnt = 0
	    fw = open("%s/predict_result_tag_%s_%s_%s.txt" % (predict_result_file, day, model_str, index), 'a+')
	    for p0, p1, u, m, c in zip(prob_0, prob_1, uid, mid, cat):
		logger.info("%s\t%s\t%s\t%s\t%s\n" % (str(p0), str(p1), str(u), str(m), str(c)))
	        fw.write("%s\t%s\t%s\t%s\t%s\n" % (str(p0), str(p1), str(u), str(m), str(c)))
	    fw.close()
	sess._finish = False
	return stored_arr
    except Exception as e:
	logger.error(str(e))


def predict_all_item_mutliprocess(predict_file=predict_file, uid_voc=uid_voc,
                                  mid_voc=mid_voc,
                                  cat_voc=cat_voc,
                                  item_info=item_info,
                                  reviews_info=reviews_info,
                                  batch_size=128,
                                  maxlen=100,
                                  day="default"):
    


    eval_sess = xdl.TrainSession()
    # print("model")
    saver = xdl.Saver()
    saver.restore(version="ckpt-................2000")


    num_pool = 4
    p = Pool(num_pool)
    file_list = os.listdir(predict_file)
    abs_file_to_predict = []
    for f in file_list:
        if f != "_SUCCESS":
            abs_file_to_predict.append(os.path.join(predict_file, f))
    #print(",".join(abs_file_to_predict))
    #print(xdl.get_config('model'))
    process = [
        p.apply_async(predict_each_core, args=(eval_sess, abs_file, predict_result_file, day, xdl.get_config('model'), uid_voc, mid_voc,
                                               cat_voc, item_info, reviews_info, maxlen, idx,))
        for idx, abs_file in enumerate(abs_file_to_predict)]
    p.close()
    p.join()


def predict_all_item(train_file=test_file,
                     test_file=test_file,
                     uid_voc=uid_voc,
                     mid_voc=mid_voc,
                     cat_voc=cat_voc,
                     item_info=item_info,
                     reviews_info=reviews_info,
                     batch_size=128,
                     maxlen=100,
                     day="default"):
    # sample_io
    if xdl.get_config('model') == 'din':
        model = Model_DIN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'dien':
        model = Model_DIEN(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien model')

    @xdl.tf_wrapper(is_training=False, gpu_memory_fraction=0.9, device_type="gpu")
    def tf_test_model(*inputs):
        with tf.variable_scope("tf_model", reuse=tf.AUTO_REUSE):
            model.build_tf_net(inputs, False)
        predict_ops = model.predict_ops()	
        return predict_ops[0], predict_ops[1:]

    eval_sess = xdl.TrainSession()
    #print("model")
    saver = xdl.Saver()
    saver.restore(version="ckpt-................5000")
    #print("predict_start")
    #print("make_SampleIO_DATA")
    sample_io = SampleIO(test_file, test_file, uid_voc, mid_voc,
                         cat_voc, item_info, reviews_info,32, maxlen, EMBEDDING_DIM)
    # predict
    #print("data_next")
    #print("SampleIO_next")
    ids, datas = sample_io.next_predict()
    #print("data_size",len(ids),len(datas))
    predict_ops = tf_test_model(*model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))  # predict_ops中包含有uuid
    #print("reslen",len(res))
    #predict_ops=[predict_ops_raw[0],predict_ops_raw[1:]]
    #idx_ops=[idx_ops_raw[0],idx_ops_raw[1],idx_ops_raw[2]]
    #print("PredictAndSave")
    #print("predict_real_start")
    stored_arr = predict_all_item_model(eval_sess, ids, predict_ops,predict_result_file, day,xdl.get_config('model'))
    cnt = 0
    #print("predict_finish")
    #fw = open("%s/predict_result_tag_%s.txt" % (predict_result_file, day), 'a+')
    #for r in stored_arr:
    #    fw.write("%s\t%s\t%s\t%s\t%s\n" % (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])))
    #    cnt += 1
    #    if cnt < 10:
    #        print(r[0], r[1], r[2], r[3], r[4])
    #fw.close()


if __name__ == '__main__':
    SEED = xdl.get_config("seed")
    if SEED is None:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    session = tf.Session(config=config)
    
    job_type = xdl.get_config("job_type")
    if job_type == 'train':
        train()
    elif job_type == 'test':
        test()
    elif job_type == "predict":
        #d = datetime.datetime.now().strftime('%Y-%m-%d')
	predict_all_item(day=yes_d)
	#predict_all_item_mutliprocess(day=yes_d)
        #predict(day=d)
    else:
        print('job type must be train or test, do nothing...')

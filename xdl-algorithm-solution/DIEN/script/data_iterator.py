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

import numpy
import json
import cPickle as pkl
import random
import gzip
import shuffle
from tensorflow.python.lib.io import file_io


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return file_io.FileIO(filename, mode)


def load_dict(filename):
    try:
        with fopen(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with fopen(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 not_predict=True):
        self.not_predict = not_predict
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        f_meta = fopen(item_info, "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:  # mid
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:  # cat
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        f_review = fopen(reviews_info, "r")
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:  # mid,item
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])
	#print(self.n_uid,self.n_mid,self.n_cat)

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):  # 总用户数，总item数，总category数
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))  # 0       ALEIRCCL8P1C4   0061284874      Books   00605794120060916575   BooksBooks

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = numpy.array(
                    [len(s[4].split('\x02')) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0  # uuid idx
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0  # mid idx
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0  # cat idx
                tmp = []
                for fea in ss[4].split('\x02'):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp  # 商品item list

                tmp1 = []
                for fea in ss[5].split('\x02'):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                # read from source file and map to word index

                # if len(mid_list) > self.maxlen:
                #    continue
                if self.skip_empty and (not mid_list):
                    continue

                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(
                            0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]  # 用于负采样，选择没有点击的item
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break  # 选择五个没有点击过的商品
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                if self.not_predict:
                    source.append([uid, mid, cat, mid_list, cat_list,
                                   noclk_mid_list, noclk_cat_list])  # [[样本信息]...]
                    target.append([float(ss[0]), 1 - float(ss[0])])  # [[target信息]...]
                else:
		    print("meta_id_map,",len(self.meta_id_map))
                    for mid_idx, cat_idx in self.meta_id_map.items():
                        source.append([uid, mid_idx, cat_idx, mid_list, cat_list,
                                       noclk_mid_list, noclk_cat_list])  # [[样本信息]...]
                        target.append([float(ss[0]), 1 - float(ss[0])])  # [[target信息]...]

                if len(source) >= self.batch_size*len(self.meta_id_map) or len(target) >= self.batch_size*len(self.meta_id_map):
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
	print(len(source),len(target))
        return source, target

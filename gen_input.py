# -*- coding:utf-8 -*-
# !/usr/bin/env python
import glob
import io

import numpy as np
import tensorflow as tf
import tensorpack as tp
from PIL import Image

import data_pb2


def parse_fn(data):
    datum = data_pb2.Datum()
    datum.ParseFromString(data[1])
    image = np.array(
        Image.open(io.BytesIO(datum.image))).astype(np.float32)
    images = np.stack(
        [np.array(Image.open(io.BytesIO(m))).astype(np.float32) for m in
         datum.images])

    label = datum.label
    labels = np.array(datum.labels)

    value = datum.value
    values = np.array(datum.values)

    string = datum.string
    strings = tuple(datum.strings)
    return tuple([image, images, label, labels, value, values, string, strings])


def train_input(path):
    paths = [f for f in glob.glob(path + '/*')]
    lmdb_dataset = [tp.LMDBData(f, True) for f in paths]
    lmdb_dataset = tp.ConcatData(lmdb_dataset)
    lmdb_dataset = tp.LocallyShuffleData(lmdb_dataset, 256)
    lmdb_dataset = tp.RepeatedData(lmdb_dataset, -1)
    lmdb_dataset = tp.MultiProcessMapDataZMQ(lmdb_dataset, 16, parse_fn)
    lmdb_dataset.reset_state()
    dataset = tf.data.Dataset.from_generator(
        lambda: tuple(lmdb_dataset.get_data()),
        (tf.float32, tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
         tf.string, tf.string))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda *x: tuple([x[:2], x[2:]]), batch_size=16,
            drop_remainder=True, num_parallel_calls=32))
    return dataset.prefetch(-1)


def eval_input(path):
    paths = [f for f in glob.glob(path + '/*')]
    lmdb_dataset = [tp.LMDBData(f, False) for f in paths]
    lmdb_dataset = tp.ConcatData(lmdb_dataset)
    lmdb_dataset = tp.RepeatedData(lmdb_dataset, 1)
    lmdb_dataset = tp.MultiThreadMapData(lmdb_dataset, 8, parse_fn, strict=True)
    lmdb_dataset = tp.MultiProcessRunnerZMQ(lmdb_dataset, 1)
    lmdb_dataset.reset_state()
    dataset = tf.data.Dataset.from_generator(
        lambda: tuple(lmdb_dataset.get_data()),
        (tf.float32, tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
         tf.string, tf.string))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda *x: tuple([x[:2], x[2:]]), batch_size=1,
            drop_remainder=False, num_parallel_calls=32))
    return dataset.prefetch(-1)

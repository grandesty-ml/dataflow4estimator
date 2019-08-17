# -*- coding:utf-8 -*-
# !/usr/bin/env python
import io
import os

import lmdb
import numpy as np
from PIL import Image

import data_pb2


def write_lmdb(filename, start=0):
    """
    write data into lmdb file
    """
    with lmdb.open(str(filename), map_size=1024 ** 4) as w:
        for i in range(start, start + 100):
            datum = data_pb2.Datum()
            image = np.random.randint(0, 255, 32 * 32 * 3,
                                      dtype=np.uint8).reshape([32, 32, 3])
            jpg_io = io.BytesIO()
            Image.fromarray(image).save(jpg_io, format='jpeg', quatily=95)
            datum.image = jpg_io.getvalue()
            png_io = io.BytesIO()
            Image.fromarray(image[:, :, 0]).save(png_io, format='png')
            datum.images.append(png_io.getvalue())
            datum.images.append(png_io.getvalue())

            datum.label = np.random.choice([0, 1, 2, 3])
            datum.labels.append(0)
            datum.labels.append(0)

            datum.value = 0.
            datum.values.append(0.)
            datum.values.append(0.)

            datum.string = '0.0'
            datum.strings.append('0.0')
            datum.strings.append('0.0')

            txn = w.begin(write=True)
            txn.put(key=str(i).encode(), value=datum.SerializeToString())
            txn.commit()


if __name__ == '__main__':
    os.mkdir('./data')
    for ii in range(3):
        file_name = './data/{}-of-3_lmdb'.format(ii)
        write_lmdb(file_name, 100 * ii)


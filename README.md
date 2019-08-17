# Dataflow for tensorflow Estimator

This is a test of tensorpack.dataflow for tensorflow.estimator

## Requirements

- python 3.6
- tensorflow 1.13.1
- tensorpack 0.9.7.0

## Running

1. compile the proto file:

```bash
protoc data.proto --python_out=.
```

2. generate the lmdb dataset:

```bash
python gen_data.py
```

3. train the model:

```bash
python train.py
```

## Problem

The training program would always be going to hang, and then it occupies all the memory gradually.
# -*- coding:utf-8 -*-
# !/usr/bin/env python
import os

import tensorflow as tf

from gen_input import eval_input
from gen_input import train_input
from model import model_fn

gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.1,
    allow_growth=True,
    visible_device_list='0')
config = tf.ConfigProto(
    log_device_placement=False, gpu_options=gpu_options)
run_config = tf.estimator.RunConfig(
    session_config=config,
    model_dir='./model/',
    save_summary_steps=50,
    save_checkpoints_steps=100,
    log_step_count_steps=10,
    keep_checkpoint_max=10)

estimator = tf.estimator.Estimator(
    model_fn=model_fn, config=run_config, params=None)
train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input('./data'), max_steps=1000)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: eval_input('./data'), steps=None, throttle_secs=10)
tf.logging.set_verbosity(tf.logging.INFO)
os.mkdir('./model')
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

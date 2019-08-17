# -*- coding:utf-8 -*-
# !/usr/bin/env python
import tensorflow as tf


def gen_model():
    """gen a simple model"""
    input_img = tf.keras.Input([32, 32, 3])
    x = input_img
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.AvgPool2D(strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.AvgPool2D(strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    logit = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inputs=input_img, outputs=logit)


def model_fn(features, labels, mode, params=None):
    model = gen_model()
    logit = model(features[0])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        tf.one_hot(labels[0], 4), logit))
    if model.losses:
        for o in model.losses:
            loss += tf.reduce_sum(o)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        training_optimizer = tf.train.AdamOptimizer()
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_or_create_global_step(),
            learning_rate=None,
            clip_gradients=10.,
            optimizer=training_optimizer,
            update_ops=model.get_updates_for(model.inputs),
            name='')
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        prob = tf.nn.softmax(logit)
        predicted = tf.argmax(prob)
        accuracy = tf.metrics.accuracy(labels[0], predicted)
        eval_metric_ops = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    raise ValueError()

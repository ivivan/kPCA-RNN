import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.estimator.python.estimator import head,multi_head
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
import warnings

warnings.filterwarnings("ignore")


def lstm_model(features, labels, mode, params): # [Ftrl, Adam, Adagrad, Momentum, SGD, RMSProp]
    """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicRNNCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [rnn.DropoutWrapper(rnn.BasicRNNCell(layer['num_units']),layer['keep_prob'])
                    if layer.get('keep_prob')
                    else rnn.BasicRNNCell(layer['num_units'])
                    for layer in layers]

        return [rnn.BasicRNNCell(steps) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers


    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(params['RNN_LAYERS']), state_is_tuple=True)
    x_ = tf.unstack(features['x'], num=params['TIMESTEPS'], axis=1)
    output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
    output = dnn_layers(output[-1], params['DENSE_LAYERS'])

    # regression layer for 1 dimentional output
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # three different output layers for three predictions
    # prediction_ntu = tf.contrib.layers.fully_connected(output,1,None)
    # prediction_ec = tf.contrib.layers.fully_connected(output,1,None)
    # prediction_temp = tf.contrib.layers.fully_connected(output,1,None)



    # # Create simple heads and specify head name.
    # # label_dimension??
    # head1 = head.regression_head(weight_column=None, label_dimension=1, name='y_train_ntu')
    # head2 = head.regression_head(weight_column=None, label_dimension=1, name='y_train_ec')
    # head3 = head.regression_head(weight_column=None, label_dimension=1, name='y_train_temp')
    #
    # # Create multi-head from two simple heads.
    # allhead = multi_head.multi_head([head1, head2,head3])
    # # Create logits for each head, and combine them into a dict.
    # logits1, logits2,logits3 = multi_head.logit_fn()
    # logits = {'y_train_ntu': logits1, 'y_train_ec': logits2,'y_train_temp':logits3}
    #
    #
    # def _train_op_fn(loss):
    #     return tf.train.Optimizer.minimize(loss)
    #
    # # Return the merged EstimatorSpec
    # return head.create_estimator_spec(features=features,mode=mode,logits=logits,predictions=predictions,labels=labels,train_op_fn=_train_op_fn)




    # predictions = {
    #     'prediction_ntu':prediction_ntu,
    #     'prediction_ec':prediction_ec,
    #     'prediction_temp':prediction_temp
    # }

    #Provide an estimator spec for 'ModeKeys.PREDICT'.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    #Loss for three different predictions
    #wait for tensorflow support lables as dic!!!!!!!!!!!!

    #
    # loss_ntu = tf.losses.mean_squared_error(predictions['prediction_ntu'], labels['y_train_ntu'])
    # loss_ec = tf.losses.mean_squared_error(predictions['prediction_ec'],labels['y_train_ec'])
    # loss_temp = tf.losses.mean_squared_error(predictions['prediction_temp'],labels['y_train_temp'])
    #


    # loss = loss_ntu + loss_ec + loss_temp

    loss = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=tf.train.exponential_decay(0.01, tf.contrib.framework.get_global_step(), decay_steps = 1000, decay_rate = 0.9, staircase=False, name=None))
                                             # learning_rate=0.01)

    # eval_metric_ops = {
    #     "accuracy": tf.metrics.accuracy(labels['y_train_ntu'],predictions['y_train_ntu'])
    #
    # }

    # return EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,predictions=predictions,train_op=train_op,eval_metric_ops=None)






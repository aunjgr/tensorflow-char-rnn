# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from six.moves import xrange


def layer_norm(input_tensor, num_variables_in_tensor=1, initial_bias_value=0.0, scope="layer_norm"):
  with tf.variable_scope(scope):
    '''for clarification of shapes:
    input_tensor = [batch_size, num_neurons]
    mean = [batch_size]
    variance = [batch_size]
    alpha = [num_neurons]
    bias = [num_neurons]
    output = [batch_size, num_neurons]
    '''
    input_tensor_shape_list = input_tensor.get_shape().as_list()
    num_neurons = input_tensor_shape_list[1] // num_variables_in_tensor

    alpha = tf.get_variable('layer_norm_alpha', [num_neurons * num_variables_in_tensor],
                            initializer = tf.constant_initializer(1.0))
    bias = tf.get_variable('layer_norm_bias', [num_neurons * num_variables_in_tensor],
                           initializer = tf.constant_initializer(initial_bias_value))

    if num_variables_in_tensor == 1:
      input_tensor_list = [input_tensor]
      alpha_list = [alpha]
      bias_list = [bias]
    else:
      input_tensor_list = tf.split(input_tensor, axis=1, num_or_size_splits=num_variables_in_tensor)
      alpha_list = tf.split(alpha, axis=0, num_or_size_splits=num_variables_in_tensor)
      bias_list = tf.split(bias, axis=0, num_or_size_splits=num_variables_in_tensor)

    list_of_layer_normed_results = []
    for counter in xrange(num_variables_in_tensor):
      mean, variance = moments_for_layer_norm(input_tensor_list[counter], axes=[1], name="moments_loopnum_" + str(counter) + scope) #average across layer
      output =  (alpha_list[counter] * (input_tensor_list[counter] - mean)) / variance + bias[counter]
      list_of_layer_normed_results.append(output)

    if num_variables_in_tensor == 1:
      return list_of_layer_normed_results[0]
    else:
      return tf.concat(list_of_layer_normed_results, 1)


def moments_for_layer_norm(x, axes=1, name=None, epsilon=0.001):
  '''output for mean and variance should be [batch_size]'''

  if not isinstance(axes, list): axes = list(axes)

  with tf.name_scope(values=[x, axes], name=name, default_name="moments"):
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)

    return mean, variance

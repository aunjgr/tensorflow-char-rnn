#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import contextlib
import tensorflow as tf
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"


@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
    if reuse is not None:
        kwargs["reuse"] = reuse
    with tf.variable_scope(scope, **kwargs) as checking_scope:
        scope_name = checking_scope.name
        if hasattr(cell, "_scope"):
            cell_scope = cell._scope  # pylint: disable=protected-access
            if cell_scope.name != checking_scope.name:
                raise ValueError(
                    "Attempt to reuse RNNCell %s with a different variable scope than "
                    "its first use.  First use of cell was with scope '%s', this "
                    "attempt is with scope '%s'.  Please create a new instance of the "
                    "cell if you would like it to use a different set of weights.  "
                    "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
                    "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
                    "If before you were using the same cell instance as both the "
                    "forward and reverse cell of a bidirectional RNN, simply create "
                    "two instances (one for forward, one for reverse).  "
                    "In May 2017, we will start transitioning this cell's behavior "
                    "to use existing stored weights, if any, when it is called "
                    "with scope=None (which can lead to silent model degradation, so "
                    "this error will remain until then.)"
                    % (cell, cell_scope.name, scope_name, type(cell).__name__,
                       type(cell).__name__))
        else:
            weights_found = False
            try:
                with tf.variable_scope(checking_scope, reuse=True):
                    tf.get_variable(_WEIGHTS_VARIABLE_NAME)
                weights_found = True
            except ValueError:
                pass
            if weights_found and reuse is None:
                raise ValueError(
                    "Attempt to have a second RNNCell use the weights of a variable "
                    "scope that already has weights: '%s'; and the cell was not "
                    "constructed as %s(..., reuse=True).  "
                    "To share the weights of an RNNCell, simply "
                    "reuse it in your second calculation, or create a new one with "
                    "the argument reuse=True." % (scope_name, type(cell).__name__))

        # Everything is OK.  Update the cell's scope and yield it.
        cell._scope = checking_scope  # pylint: disable=protected-access
        yield checking_scope


def _linear(args, output_size, bias, bias_start=0.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
        return tf.nn.bias_add(res, biases)


class MultiplicativeLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with _checked_scope(self, scope or "mlstm_cell", reuse=self._reuse):
            c, h = state

            with tf.variable_scope("mx"):
                mx = _linear([inputs], self._num_units, False)
            with tf.variable_scope("mh"):
                mh = _linear([h], self._num_units, False)
            m = mx * mh

            with tf.variable_scope("gates"):
                z = _linear([inputs, m], 4 * self._num_units, True)
            i, f, o, u = tf.split(value=z, axis=1, num_or_size_splits=4)
            c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(u)
            h = tf.nn.sigmoid(o) * self._activation(c)

        return h, tf.contrib.rnn.LSTMStateTuple(c, h)


class MultiplicativeGRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with _checked_scope(self, scope or "mgru_cell", reuse=self._reuse):
            h = state
            with tf.variable_scope("mx"):
                mx = _linear([inputs], self._num_units, False)
            with tf.variable_scope("mh"):
                mh = _linear([h], self._num_units, False)
            m = mx * mh

            with tf.variable_scope("gates"):
                z = tf.sigmoid(_linear([inputs, m], 2 * self._num_units, True, 1.0))
            r, u = tf.split(value=z, axis=1, num_or_size_splits=2)

            with tf.variable_scope("candidate"):
                c = self._activation(_linear([inputs, r * h], self._num_units, True))
            h = u * h + (1 - u) * c

        return h, h

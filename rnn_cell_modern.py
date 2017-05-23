"""Module for constructing RNN Cells"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
import contextlib
from six.moves import xrange
import tensorflow as tf

# from multiplicative_integration import multiplicative_integration, multiplicative_integration_for_multiple_inputs

import highway_network_modern
from multiplicative_integration_modern import multiplicative_integration
from normalization_ops_modern import layer_norm

from linear_modern import linear

RNNCell = tf.contrib.rnn.RNNCell


class HighwayRNNCell(RNNCell):
  """Highway RNN Network with multiplicative_integration"""

  def __init__(self, num_units, num_highway_layers = 3, use_inputs_on_each_layer = False):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state
    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          highway_factor = tf.tanh(linear([inputs, current_state], self._num_units, True))
        else:
          highway_factor = tf.tanh(linear([current_state], self._num_units, True))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))
        else:
          gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

        gate_for_hidden_factor = 1.0 - gate_for_highway_factor

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

    return current_state, current_state

class BasicGatedCell(RNNCell):
  """Basic Gated Cell from NasenSpray on reddit: https://www.reddit.com/r/MachineLearning/comments/4vyv89/minimal_gate_unit_for_recurrent_neural_networks/"""

  def __init__(self, num_units, use_multiplicative_integration = True,
    use_recurrent_dropout = False, recurrent_dropout_factor = 0.90, is_training = True,
    forget_bias_initialization = 1.0):
    self._num_units = num_units
    self.use_multiplicative_integration = use_multiplicative_integration
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_factor = recurrent_dropout_factor
    self.is_training = is_training
    self.forget_bias_initialization = forget_bias_initialization

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0,scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      with tf.variable_scope("Gates"):  # Forget Gate bias starts as 1.0 -- TODO: double check if this is correct
        if self.use_multiplicative_integration:
          gated_factor = multiplicative_integration([inputs, state], self._num_units, self.forget_bias_initialization)
        else:
          gated_factor = linear([inputs, state], self._num_units, True, self.forget_bias_initialization)

        gated_factor = tf.sigmoid(gated_factor)

      with tf.variable_scope("Candidate"):
        c = tf.tanh(linear([inputs], self._num_units, True, 0.0))

        if self.use_recurrent_dropout and self.is_training:
          input_contribution = tf.nn.dropout(c, self.recurrent_dropout_factor)
        else:
          input_contribution = c

      new_h = (1 - gated_factor)*state + gated_factor * input_contribution

    return new_h, new_h


class MGUCell(RNNCell):
  """Minimal Gated Unit from  http://arxiv.org/pdf/1603.09420v1.pdf."""

  def __init__(self, num_units, use_multiplicative_integration = True,
    use_recurrent_dropout = False, recurrent_dropout_factor = 0.90, is_training = True,
    forget_bias_initialization = 1.0):
    self._num_units = num_units
    self.use_multiplicative_integration = use_multiplicative_integration
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_factor = recurrent_dropout_factor
    self.is_training = is_training
    self.forget_bias_initialization = forget_bias_initialization

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0,scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      with tf.variable_scope("Gates"):  # Forget Gate bias starts as 1.0 -- TODO: double check if this is correct
        if self.use_multiplicative_integration:
          gated_factor = multiplicative_integration([inputs, state], self._num_units, self.forget_bias_initialization)
        else:
          gated_factor = linear([inputs, state], self._num_units, True, self.forget_bias_initialization)

        gated_factor = tf.sigmoid(gated_factor)

      with tf.variable_scope("Candidate"):
        if self.use_multiplicative_integration:
          c = tf.tanh(multiplicative_integration([inputs, state*gated_factor], self._num_units, 0.0))
        else:
          c = tf.tanh(linear([inputs, state*gated_factor], self._num_units, True, 0.0))

        if self.use_recurrent_dropout and self.is_training:
          input_contribution = tf.nn.dropout(c, self.recurrent_dropout_factor)
        else:
          input_contribution = c

      new_h = (1 - gated_factor)*state + gated_factor * input_contribution

    return new_h, new_h

class LSTMCell_MemoryArray(RNNCell):
  """Implementation of Recurrent Memory Array Structures Kamil Rocki
  https://arxiv.org/abs/1607.03085

  Idea is to build more complex memory structures within one single layer rather than stacking multiple layers of RNNs

  """

  def __init__(self, num_units, num_memory_arrays = 2, use_multiplicative_integration = True, use_recurrent_dropout = False, recurrent_dropout_factor = 0.90, is_training = True, forget_bias = 1.0,
    use_layer_normalization = False):
    self._num_units = num_units
    self.num_memory_arrays = num_memory_arrays
    self.use_multiplicative_integration = use_multiplicative_integration
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_factor = recurrent_dropout_factor
    self.is_training = is_training
    self.use_layer_normalization = use_layer_normalization
    self._forget_bias = forget_bias


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units * (self.num_memory_arrays + 1)

  def __call__(self, inputs, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      hidden_state_plus_c_list = tf.split(state, axis=1, num_or_size_splits=self.num_memory_arrays + 1)

      h = hidden_state_plus_c_list[0]
      c_list = hidden_state_plus_c_list[1:]

      '''very large matrix multiplication to speed up procedure -- will split variables out later'''

      if self.use_multiplicative_integration:
        concat = multiplicative_integration([inputs, h], self._num_units * 4 * self.num_memory_arrays, 0.0)
      else:
        concat = linear([inputs, h], self._num_units * 4 * self.num_memory_arrays, True)

      if self.use_layer_normalization: concat = layer_norm(concat, num_variables_in_tensor = 4 * self.num_memory_arrays)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate -- comes in sets of fours
      all_vars_list = tf.split(concat, axis=1, num_or_size_splits=4 * self.num_memory_arrays)

      '''memory array loop'''
      new_c_list, new_h_list = [], []
      for array_counter in xrange(self.num_memory_arrays):

        i = all_vars_list[0 + array_counter * 4]
        j = all_vars_list[1 + array_counter * 4]
        f = all_vars_list[2 + array_counter * 4]
        o = all_vars_list[3 + array_counter * 4]

        if self.use_recurrent_dropout and self.is_training:
          input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
        else:
          input_contribution = tf.tanh(j)

        new_c_list.append(c_list[array_counter] * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution)

        if self.use_layer_normalization:
          new_c = layer_norm(new_c_list[-1])
        else:
          new_c = new_c_list[-1]

        new_h_list.append(tf.tanh(new_c) * tf.sigmoid(o))

      '''sum all new_h components -- could instead do a mean -- but investigate that later'''
      new_h = tf.add_n(new_h_list)

    return new_h, tf.concat(1, [new_h] + new_c_list) #purposely reversed








class JZS1Cell(RNNCell):
  """Mutant 1 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS1, mutant 1 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1 z = sigm(WxzXt+Bz), x_t is inputs'''

          z = tf.sigmoid(linear([inputs],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))

        with tf.variable_scope("Rinput"):
          '''equation 2 r = sigm(WxrXt+Whrht+Br), h_t is the previous state'''

          r = tf.sigmoid(linear([inputs,state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear([r*state],
                            self._num_units, True)
          component_1 = tf.tanh(tf.tanh(inputs) + component_0)
          component_2 = component_1*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of.
      #This makes it more mem efficient than LSTM


class JZS2Cell(RNNCell):
  """Mutant 2 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS2, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          '''equation 1'''

          z = tf.sigmoid(linear([inputs, state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))

          '''equation 2 '''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(inputs+(linear([state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor)))
          '''equation 3'''

        with tf.variable_scope("Candidate"):

          component_0 = linear([state*r,inputs],
                            self._num_units, True)

          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of.
        #This makes it more mem efficient than LSTM

class JZS3Cell(RNNCell):
  """Mutant 3 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS3, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1'''

          z = tf.sigmoid(linear([inputs, tf.tanh(state)],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))

          '''equation 2'''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(linear([inputs, state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear([state*r,inputs],
                            self._num_units, True)

          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of.
      #This makes it more mem efficient than LSTM



class Delta_RNN(RNNCell):
  """
  From https://arxiv.org/pdf/1703.08864.pdf

  Implements a second order Delta RNN with inner and outer functions
  """

  def __init__(self, num_units, activation=tf.nn.elu, reuse=None):
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def _outer_function(self, inner_function_output, past_hidden_state, wx_parameterization_gate=True):
    """Simulates Equation 3 in Delta RNN paper

    r, the gate, can be parameterized in many different ways.
    """

    #assert inner_function_output.get_shape().as_list() == past_hidden_state.get_shape().as_list()
    r = tf.get_variable("outer_function_gate", [self._num_units], dtype=tf.float32, initializer=tf.zeros_initializer)

    # Equation 5 in Delta Rnn Paper
    if wx_parameterization_gate:
      r = self._W_x_inputs + r

    gate = tf.sigmoid(r)
    output = self._activation((1.0 - gate) * inner_function_output + gate * past_hidden_state)

    return output

  def _inner_function(self, inputs, past_hidden_state, activation=tf.tanh):
    """second order function as described equation 11 in delta rnn paper
    The main goal is to produce z_t of this function
    """
    with tf.variable_scope("V_x"):
      V_x_d = linear([past_hidden_state], self._num_units, True)

    # We make this a private variable to be reused in the _outer_function
    with tf.variable_scope("W_x"):
      self._W_x_inputs = linear([inputs], self._num_units, True)

    alpha = tf.get_variable("alpha", [self._num_units], dtype=tf.float32, initializer=tf.ones_initializer)
    beta_one = tf.get_variable("beta_one", [self._num_units], dtype=tf.float32, initializer=tf.ones_initializer)
    beta_two = tf.get_variable("beta_two", [self._num_units], dtype=tf.float32, initializer=tf.ones_initializer)
    z_t_bias = tf.get_variable("z_t_bias", [self._num_units], dtype=tf.float32, initializer=tf.zeros_initializer)

    # Second Order Cell Calculations
    d_1_t = alpha * V_x_d * self._W_x_inputs
    d_2_t = beta_one * V_x_d + beta_two * self._W_x_inputs
    z_t = activation(d_1_t + d_2_t + z_t_bias)

    return z_t

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "delta_rnn_cell", reuse=self._reuse):
      output = self._outer_function(self._inner_function(inputs, state), state)

    return output, output #there is only one hidden state output to keep track of.


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

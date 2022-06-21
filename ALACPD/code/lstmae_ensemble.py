
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, RNN, Lambda
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, GRU, Conv2D, Dropout, Flatten, Dense, Reshape, Concatenate, Add, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
import numpy as np 


from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export

class DropoutRNNCellMixin(object):
  """Object that hold dropout related fields for RNN Cell.
  This class is not a standalone RNN cell. It suppose to be used with a RNN cell
  by multiple inheritance. Any cell that mix with class should have following
  fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

  def __init__(self, *args, **kwargs):
    # Note that the following two masks will be used in "graph function" mode,
    # e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    # tensors will be generated differently than in the "graph function" case,
    # and they will be cached.
    # Also note that in graph mode, we still cache those masks only because the
    # RNN could be created with `unroll=True`. In that case, the `cell.call()`
    # function will be invoked multiple times, and we want to ensure same mask
    # is used every time.
    self._dropout_mask = None
    self._recurrent_dropout_mask = None
    self._eager_dropout_mask = None
    self._eager_recurrent_dropout_mask = None
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  def reset_dropout_mask(self):
    """Reset the cached dropout masks if any.
    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._dropout_mask = None
    self._eager_dropout_mask = None

  def reset_recurrent_dropout_mask(self):
    """Reset the cached recurrent dropout masks if any.
    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._recurrent_dropout_mask = None
    self._eager_recurrent_dropout_mask = None

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the dropout mask for RNN cell's input.
    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.
    Args:
      inputs: the input tensor whose shape will be used to generate dropout
        mask.
      training: boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
      return None
    if (not context.executing_eagerly() and self._dropout_mask is None
        or context.executing_eagerly() and self._eager_dropout_mask is None):
      # Generate new mask and cache it based on context.
      dp_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.dropout,
          training=training,
          count=count)
      if context.executing_eagerly():
        self._eager_dropout_mask = dp_mask
      else:
        self._dropout_mask = dp_mask
    else:
      # Reuse the existing mask.
      dp_mask = (self._eager_dropout_mask
                 if context.executing_eagerly() else self._dropout_mask)
    return dp_mask

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the recurrent dropout mask for RNN cell.
    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.
    Args:
      inputs: the input tensor whose shape will be used to generate dropout
        mask.
      training: boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.recurrent_dropout == 0:
      return None
    if (not context.executing_eagerly() and self._recurrent_dropout_mask is None
        or context.executing_eagerly()
        and self._eager_recurrent_dropout_mask is None):
      # Generate new mask and cache it based on context.
      rec_dp_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.recurrent_dropout,
          training=training,
          count=count)
      if context.executing_eagerly():
        self._eager_recurrent_dropout_mask = rec_dp_mask
      else:
        self._recurrent_dropout_mask = rec_dp_mask
    else:
      # Reuse the existing mask.
      rec_dp_mask = (self._eager_recurrent_dropout_mask
                     if context.executing_eagerly()
                     else self._recurrent_dropout_mask)
    return rec_dp_mask
#######################################################################################################################
#                                      Start AR specific layers subsclass                                             #
#                                                                                                                     #
# The AR layer is implemented as follows:                                                                             #
# - Pre Transformation layer that takes a 'highway' size of data and apply a reshape and axis transformation          #
# - Flatten the output and pass it through a Dense layer with one output                                              #
# - Post Transformation layer that bring back dimensions to its original shape                                        #                                                     #
#######################################################################################################################

class PreARTrans(tf.keras.layers.Layer):
    def __init__(self, hw, **kwargs):
        #
        # hw: Highway = Number of timeseries values to consider for the linear layer (AR layer)
        #
        self.hw = hw
        super(PreARTrans, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PreARTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors; in this case it's just one tensor: X = the input to the model
        x = inputs

        # Get the batchsize which is tf.shape(x)[0]
        batchsize = tf.shape(x)[0]

        # Get the shape of the input data
        input_shape = K.int_shape(x)
        
        # Select only 'highway' length of input to create output
        output = x[:,-self.hw:,:]
        
        # Permute axis 1 and 2. axis=2 is the the dimension having different time-series
        # This dimension should be equal to 'm' which is the number of time-series.
        output = tf.transpose(output, perm=[0,2,1])
        
        # Merge axis 0 and 1 in order to change the batch size
        output = tf.reshape(output, [batchsize * input_shape[2], self.hw])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        
        return output
    
    def compute_output_shape(self, input_shape):
        # Set the shape of axis=1 to be hw since the batchsize is NULL
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.hw
        
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'hw': self.hw}
        base_config = super(PreARTrans, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class PostARTrans(tf.keras.layers.Layer):
    def __init__(self, m, **kwargs):
        #
        # m: Number of timeseries
        #
        self.m = m
        super(PostARTrans, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PostARTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors
        # First one is the output of the Dense(1) layer which we will operate on
        # The second is the oiriginal model input tensor which we will use to get
        # the original batchsize
        x, original_model_input = inputs

        # Get the batchsize which is tf.shape(original_model_input)[0]
        batchsize = tf.shape(original_model_input)[0]

        # Get the shape of the input data
        input_shape = K.int_shape(x)
        
        # Reshape the output to have the batch size equal to the original batchsize before PreARTrans
        # and the second dimension as the number of timeseries
        output = tf.reshape(x, [batchsize, self.m])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        
        return output
    
    def compute_output_shape(self, input_shape):
        # Adjust shape[1] to be equal 'm'
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.m
        
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'m': self.m}
        base_config = super(PostARTrans, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#######################################################################################################################
#                                      End AR specific layers subsclass                                               #
#######################################################################################################################



class skip_LSTMCell(DropoutRNNCellMixin, Layer):


  def __init__(self,
               units,
               skip,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    super(skip_LSTMCell, self).__init__(**kwargs)
    self.units = units
    self.skip = skip
    
    
    
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
   
    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.implementation = implementation
    
    self.state_size = data_structures.NoDependency([self.units, self.units, 1, self.units *self.skip])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.kernel2 = self.add_weight(shape = (self.units, self.units), name="kernel2")
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    #self.count = 0#K.variable(0, name="counttttt")
    self._step = None
    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
        def bias_initializer2(_, *args, **kwargs):
          return [
              self.bias_initializer(self.units, *args, **kwargs)]
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
      self.bias2 = self.add_weight(
          shape=(self.units,),
          name='bias2',
          initializer=self.bias_initializer)
      self.s0 = self.add_weight(
          shape=(1,),
          name='s0',
          initializer= tf.keras.initializers.Constant(
    value=0.5
    ) )#self.bias_initializer)
      #self.s1 = self.add_weight(
      #    shape=(1,),
      #    name='s1',
      #    initializer=self.bias_initializer)
    else:
      self.bias = None
    self.built = True

  
  def weight_bias(self, W_shape, b_shape, bias_init=0.1):
      """Fully connected highway layer adopted from 
         https://github.com/fomorians/highway-fcn/blob/master/main.py
      """
      #with tf.variable_scope('extra', reuse = tf.AUTO_REUSE ):
      W =  tf.compat.v1.get_variable("weight_h_2", shape=W_shape,
              initializer=tf.truncated_normal_initializer(stddev=0.1))
      b =  tf.compat.v1.get_variable("bias_h_2", shape=b_shape,
              initializer=tf.truncated_normal_initializer(stddev=0.1))
      s0 =  tf.compat.v1.get_variable("s0_h_2", shape=1,
              initializer=tf.truncated_normal_initializer(stddev=0.1))
      s1 =  tf.compat.v1.get_variable("s1_h_2", shape=1,
              initializer=tf.truncated_normal_initializer(stddev=0.1))
      return W, b, s0, s1

 
        
  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    h_tm1_i = tf.cast(h_tm1_i, tf.float32)
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry stat
    step  = states[2] 
    prev_h = states[3]

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs *= dp_mask[0]
      z = K.dot(inputs, self.kernel)
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 *= rec_dp_mask[0]
      z += K.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)

      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)
    
    h = o * self.activation(c)
    

    new_h_skip = tf.sigmoid(tf.matmul(prev_h[:,:self.units], self.kernel2) + self.bias2)
    h = self.s0 * h + new_h_skip*(1- self.s0)
    #h = 0.5 * h + 0.5 * new_h_skip



    output_list = []
    prev_h = tf.roll(prev_h, shift=self.units, axis=1)
    output_list.append(tf.keras.layers.Concatenate(axis=1)([prev_h[:, :-self.units] , h[:,:]]))
    
    prev_h = tf.stack(output_list[0])
    return h, [h, c, step+1, prev_h]

 
 
  def masked_weight(self, _load=False):
        if _load==False:
            masked_W1 = np.random.randint(2, size=1)
            if masked_W1 == 0:
                masked_W2 = 1
            else:
                masked_W2 = np.random.randint(2, size=1)
  
        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(skip_LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None or dtype is None:
            raise ValueError(
                ("'batch_size and dtype cannot be None while constructing "
                  "initial state: 'batch_size={}, dtype={}'").format(
                    batch_size, dtype))
	    
        def create_zeros(unnested_state_size):
            flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
            if len(flat_dims) == 1 and flat_dims[0] == 1:
                # return array_ops.zeros(flat_dims, dtype='int64')
                return tf.constant(0, dtype='int64')
            else:
                return array_ops.zeros([batch_size] + flat_dims, 
                                        dtype=dtype)
        return nest.map_structure(create_zeros, self.state_size)
	    
  #def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
  #  return list(_generate_zero_filled_state_for_cell(
  #      self, inputs, batch_size, dtype))
    

#######################################################################################################################
#                                                 Model Start                                                         #
#                                                                                                                     #                                                                                                              #
#######################################################################################################################



def squeeze_middle2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0]* shape[1] , shape[2], shape[3] ] )
    return x3d

def squeeze_middle2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = ( in_batch, None, in_filters )
    else :
        output_shape = ( in_batch * in_rows, in_cols, in_filters )
    return output_shape





def AE_skipLSTM_AR(init, input_shape):
    # m is the number of time-series
    m = input_shape[3]
    # Get tensor shape except batchsize
    tensor_shape = input_shape[1:]

    
    X = Input(shape = tensor_shape)
    slice_X1 = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])
    slice_X2 = tf.keras.layers.Lambda(lambda x: x[:,1:, :, :])
    X1 = slice_X1(X)
    X2 = slice_X2(X)
    
    """------------------------  Encoder   --------------------------"""
    # SkipLSTM
    if init.skip > 0:
        cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)
        layer = RNN(cell)
        SE = layer(X1)
        RE = SE

    P = RepeatVector(input_shape[2])(RE)
  
    """------------------------  Decoder   --------------------------"""
    # SkipLSTM
    if init.skip > 0:
        cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)
        layer = RNN(cell, return_sequences=True)
        SD = layer(P)
        RD = SD

    # Dense layer
    RD = Flatten()(RD)
    RD = Dense(m*input_shape[2])(RD)
    Y = Reshape((input_shape[2], m))(RD)


    """------------------------  Autoregressive   --------------------------""" 
    if init.highway > 0:
        for i in range(X2.shape[1]):
            slice_X2_i = tf.keras.layers.Lambda(lambda x : x[:, i, :, :])
            Z = slice_X2_i(X2)
            Z = PreARTrans(init.highway)(Z)
            Z = Flatten()(Z)
            Z = Dense(1)(Z)
            Z = PostARTrans(m)([Z,X])
            if i == 0:
                Z2 = Reshape((1, Z.shape[1]))(Z)
            else:
                Z = Reshape((1, Z.shape[1]))(Z)
                Z2 = Concatenate(axis=1)([Z2,Z]) 
        Y = Add()([Y, Z2])
    # Generate Model
    model = Model(inputs = X, outputs = Y)
    return model


def AE_skipLSTM(init, input_shape):
    
    # m is the number of time-series
    m = input_shape[3]

    # Get tensor shape except batchsize
    tensor_shape = input_shape[1:]

    
    X = Input(shape = tensor_shape)
    slice_X1 = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])
    slice_X2 = tf.keras.layers.Lambda(lambda x: x[:,1:, :, :])
    X1 = slice_X1(X)
    X2 = slice_X2(X)

    """------------------------  Encoder   --------------------------"""
    # SkipLSTM
    if init.skip > 0:
        cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)
        layer = RNN(cell)
        RE = layer(X1)
    
    P = RepeatVector(input_shape[2])(RE)
    
    
    """------------------------  Decoder   --------------------------"""
    # SkipLSTM
    if init.skip > 0:
        cell = skip_LSTMCell(init.SkipGRUUnits, init.skip)
        layer = RNN(cell, return_sequences=True)
        RD = layer(P)
    

    # Dense layer
    RD = Flatten()(RD)
    RD = Dense(m*input_shape[2])(RD)
    Y = Reshape((input_shape[2], m))(RD)

    model = Model(inputs = X, outputs = Y)
    return model


def AR(init, input_shape):
    # m is the number of time-series
    m = input_shape[3]

    tensor_shape = input_shape[1:]

    X = Input(shape = tensor_shape)
    slice_X1 = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])
    slice_X2 = tf.keras.layers.Lambda(lambda x: x[:,1:, :, :])
    X1 = slice_X1(X)
    X2 = slice_X2(X)

    for i in range(X2.shape[1]):
        slice_X2_i = tf.keras.layers.Lambda(lambda x : x[:, i, :, :])
        Z = slice_X2_i(X2)
        Z = PreARTrans(init.highway)(Z)
        Z = Flatten()(Z)
        Z = Dense(1)(Z)
        Z = PostARTrans(m)([Z,X])
        if i == 0:
            Z2 = Reshape((1, Z.shape[1]))(Z)
        else:
            Z = Reshape((1, Z.shape[1]))(Z)
            Z2 = Concatenate(axis=1)([Z2,Z]) 

    Y = Z2
    model = Model(inputs = X, outputs = Y)
    return model






















#
# A function that compiles 'model' after setting the appropriate:
# - optimiser function passed via init 
# - learning rate passed via init
# - loss function also set in init
# - metrics
#
def ModelCompile(model, init):
    # Select the appropriate optimiser and set the learning rate from input values (arguments)
    if init.optimiser == "SGD":
        opt = SGD(lr=init.lr, momentum=0.0, decay=0.0, nesterov=False)
    elif init.optimiser == "RMSprop":
        opt = RMSprop(lr=init.lr, rho=0.9, epsilon=None, decay=0.0)
    else: # Adam
    	opt  = Adam(lr=init.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile using the previously defined metrics
    model.compile(optimizer = opt, loss = init.loss)

    # Launch Tensorboard if selected in arguments
    if init.tensorboard != None:
        tensorboard = TensorBoard(log_dir=init.tensorboard, histogram_freq=1, write_graph=True, write_images=True)
    else:
        tensorboard = None

    return tensorboard


@keras_export(v1=['keras.layers.LSTM'])
class LSTM(RNN):
 
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      print('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation)
    super(LSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell.reset_dropout_mask()
    self.cell.reset_recurrent_dropout_mask()
    return super(LSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
  def dropped_inputs():
    return K.dropout(ones, rate)

  if count > 1:
    return [
        K.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  return K.in_train_phase(dropped_inputs, ones, training=training)


def _standardize_args(inputs, initial_state, constants, num_constants):
  """Standardizes `__call__` to a single list of tensor inputs.
  When running a model loaded from a file, the input tensors
  `initial_state` and `constants` can be passed to `RNN.__call__()` as part
  of `inputs` instead of by the dedicated keyword arguments. This method
  makes sure the arguments are separated and that `initial_state` and
  `constants` are lists of tensors (or None).
  Arguments:
    inputs: Tensor or list/tuple of tensors. which may include constants
      and initial states. In that case `num_constant` must be specified.
    initial_state: Tensor or list of tensors or None, initial states.
    constants: Tensor or list of tensors or None, constant tensors.
    num_constants: Expected number of constants (if constants are passed as
      part of the `inputs` list.
  Returns:
    inputs: Single tensor or tuple of tensors.
    initial_state: List of tensors or None.
    constants: List of tensors or None.
  """
  if isinstance(inputs, list):
    # There are several situations here:
    # In the graph mode, __call__ will be only called once. The initial_state
    # and constants could be in inputs (from file loading).
    # In the eager mode, __call__ will be called twice, once during
    # rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
    # model.fit/train_on_batch/predict with real np data. In the second case,
    # the inputs will contain initial_state and constants as eager tensor.
    #
    # For either case, the real input is the first item in the list, which
    # could be a nested structure itself. Then followed by initial_states, which
    # could be a list of items, or list of list if the initial_state is complex
    # structure, and finally followed by constants which is a flat list.
    assert initial_state is None and constants is None
    if num_constants is not None:
      constants = inputs[-num_constants:]
      inputs = inputs[:-num_constants]
    if len(inputs) > 1:
      initial_state = inputs[1:]
      inputs = inputs[:1]

    if len(inputs) > 1:
      inputs = tuple(inputs)
    else:
      inputs = inputs[0]

  def to_list_or_none(x):
    if x is None or isinstance(x, list):
      return x
    if isinstance(x, tuple):
      return list(x)
    return [x]

  initial_state = to_list_or_none(initial_state)
  constants = to_list_or_none(constants)

  return inputs, initial_state, constants


def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)



import copy
import sys

def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
    

    #prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    #prints = print_message(str(x2[0]))
    prints = tf.print(tensors[0]) #[tf.py_func(print_message, [tensor], None) for tensor in tensors]
    with tf.control_dependencies([prints]):
        op = tf.identity(op)
    return op






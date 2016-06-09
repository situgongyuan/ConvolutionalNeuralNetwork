import numpy as np
from cs231n.fast_layers import *
from time import time


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  x_flat = x.reshape(x.shape[0],-1)
  out = np.dot(x_flat,w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx_flat = np.dot(dout,w.T)
  dx = dx_flat.reshape(x.shape)
  x_flat = x.reshape(x.shape[0],-1)
  dw = np.dot(x_flat.T,dout)
  db = np.sum(dout, axis = 0)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dout[x <= 0] = 0
  dx = dout

  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    mean = np.mean(x,axis = 0)
    var = np.sum(((x - mean) ** 2), axis = 0) / N
    x_hat = (x - mean) / np.sqrt(var + eps)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    out = x_hat * gamma + beta
    cache = (x,x_hat,mean,var,gamma,eps)
  elif mode == 'test':
    mean = running_mean
    var = running_var
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = x_hat * gamma + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache
  """


  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    # Compute output
    mu = x.mean(axis=0)
    xc = x - mu
    var = np.mean(xc ** 2, axis=0)
    std = np.sqrt(var + eps)
    xn = xc / std
    out = gamma * xn + beta

    cache = (mode, x, gamma, xc, std, xn, out)

    # Update running average of mean
    running_mean *= momentum
    running_mean += (1 - momentum) * mu

    # Update running average of variance
    running_var *= momentum
    running_var += (1 - momentum) * var
  elif mode == 'test':
    # Using running mean and variance to normalize
    std = np.sqrt(running_var + eps)
    xn = (x - running_mean) / std
    out = gamma * xn + beta
    cache = (mode, x, xn, gamma, beta, std)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache



def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """

  """
  x,x_hat,mean,var,gamma,eps = cache

  N = x.shape[0]
  dgamma = np.sum(dout * x_hat, axis = 0)
  dbeta = np.sum(dout, axis = 0)

  dx_hat = dout * gamma
  partial_x = dx_hat / np.sqrt(var + eps)
  partial_var = -0.5 * (1 / np.sqrt(var + eps) ** 3) * np.sum((x - mean) * dx_hat, axis = 0)
  partial_mean = np.sum(-dx_hat / np.sqrt(var + eps), axis = 0) + partial_var / N * np.sum(-2 * (x - mean), axis = 0)

  dx = partial_x + partial_var * 2 * (x - mean) / N + partial_mean / N

  return dx, dgamma, dbeta
  """


  mode = cache[0]
  if mode == 'train':
    mode, x, gamma, xc, std, xn, out = cache

    N = x.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dxc = dxn / std
    dstd = -np.sum((dxn * xc) / (std * std), axis=0)
    dvar = 0.5 * dstd / std
    dxc += (2.0 / N) * xc * dvar
    dmu = np.sum(dxc, axis=0)
    dx = dxc - dmu / N
  elif mode == 'test':
    mode, x, xn, gamma, beta, std = cache
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dx = dxn / std
  else:
    raise ValueError(mode)

  return dx, dgamma, dbeta




def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = np.random.binomial(1,1 - p,x.shape)
    out = x * mask

  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  # Check dimensions
  assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
  assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

  # Create output
  out_height = (H + 2 * pad - filter_height) / stride + 1
  out_width = (W + 2 * pad - filter_width) / stride + 1

  x_cols = im22col_cython(x, w.shape[2], w.shape[3], pad, stride) #(10000 * 1024,27)

  out = x_cols.dot(w.reshape((w.shape[0],-1)).T) + b              #(10000 * 1024,4)
  out = out.reshape(N,out_height,out_width,-1)
  out = out.transpose(0,3,1,2)

  cache = (x, w, b, conv_param, x_cols)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param, x_cols = cache
  N, C, H, W = x.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  db = np.sum(dout,axis = (0,2,3))

  num_filters, c, filter_height, filter_width = w.shape
  dout_reshape = dout.reshape(N,num_filters,-1).transpose(0,2,1)   #(10000,1024,4)
  dout_reshape = dout_reshape.reshape(-1,num_filters)  #(10000 * 1024,4)
  dw = (dout_reshape.T).dot(x_cols).reshape(w.shape)

  dx_cols = dout_reshape.dot(w.reshape(num_filters,-1))
  dx = col22im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                     filter_height, filter_width, pad, stride)
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  same_size = pool_height == pool_width == stride
  tiles = H % pool_height == 0 and W % pool_width == 0
  if same_size and tiles:
    out, reshape_cache = max_pool_forward_reshape(x, pool_param)
    cache = ('reshape', reshape_cache)
  else:
    out, im22col_cache = max_pool_forward_im22col(x, pool_param)
    cache = ('im22col', im22col_cache)
  return out, cache


def max_pool_forward_reshape(x, pool_param):
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  assert pool_height == pool_width == stride, 'Invalid pool params'
  assert H % pool_height == 0
  assert W % pool_height == 0
  x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                         W / pool_width, pool_width)
  out = x_reshaped.max(axis=5).max(axis=3)

  cache = (x, x_reshaped, out)
  return out, cache


def max_pool_forward_im22col(x, pool_param):
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  assert (H - pool_height) % stride == 0, 'Invalid height'
  assert (W - pool_width) % stride == 0, 'Invalid width'

  out_height = (H - pool_height) / stride + 1
  out_width = (W - pool_width) / stride + 1

  x_split = x.reshape(N * C, 1, H, W)
  x_cols = im22col_cython(x_split, pool_height, pool_width, 0, stride) #(10000 * 1024 * 3, 9)
  x_cols_argmax = np.argmax(x_cols,axis = 1)
  x_cols_max = x_cols[np.arange(x_cols.shape[0]),x_cols_argmax]  #(10000 * 3 * 1024)
  out = x_cols_max.reshape(N,C,out_height,out_width)

  cache = (x, x_cols, x_cols_argmax, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  method, real_cache = cache
  if method == 'reshape':
    return max_pool_backward_reshape(dout, real_cache)
  elif method == 'im2col':
    return max_pool_backward_im22col(dout, real_cache)
  else:
    raise ValueError('Unrecognized method "%s"' % method)


def max_pool_backward_reshape(dout, cache):
  x, x_reshaped, out = cache
  x_reshaped = x_reshaped.transpose(0,1,2,4,3,5) #(100,3,16,16,2,2)
  out_newaxis = out[:, :, :, :, np.newaxis, np.newaxis]  #(100,3,16,16,1,1)
  mask = (x_reshaped == out_newaxis) #(100,3,16,16,2,2)
  dx_reshaped = mask * dout[:, :, :, :, np.newaxis,np.newaxis]
  dx = dx_reshaped.transpose(0,1,2,4,3,5).reshape(x.shape)
  return dx


def max_pool_backward_im22col(dout, cache):
  x, x_cols, x_cols_argmax, pool_param = cache
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  dx_cols = np.zeros(x_cols.shape)
  dx_cols[np.arange(x_cols.shape[0]),x_cols_argmax] = dout.flatten()
  dx = col22im_cython(dx_cols,N,C,H,W,pool_height,pool_width,0,stride)
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  N,C,H,W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  running_mean = bn_param.get('running_mean', np.zeros((H , W), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((H , W), dtype=x.dtype))

  if mode == 'train':
    mean = np.mean(x,axis = 0)
    var = np.mean((x - mean) ** 2,axis = 0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    out = x_hat * gamma.reshape(C,1,1) + beta.reshape(C,1,1)
    cache = (x,x_hat,mean,var,gamma,eps)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
  elif mode == 'test':
    mean = running_mean
    var = running_var
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = x_hat * gamma.reshape(C,1,1) + beta.reshape(C,1,1)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """

  x,x_hat,mean,var,gamma,eps = cache
  dgamma = np.sum(dout * x_hat, axis = (0,2,3))
  dbeta = np.sum(dout, axis = (0,2,3))
  N,C,H,W = x.shape

  dx_hat = dout * gamma.reshape(C,1,1)
  partial_x = dx_hat / np.sqrt(var + eps)
  partial_var = -0.5 * (1 / np.sqrt(var + eps) ** 3) * np.sum((x - mean) * dx_hat, axis = 0)
  partial_mean = np.sum(-dx_hat / np.sqrt(var + eps), axis = 0) + partial_var / N * np.sum(-2 * (x - mean), axis = 0)

  dx = partial_x + partial_var * 2 * (x - mean) / N + partial_mean / N
  return dx, dgamma, dbeta

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

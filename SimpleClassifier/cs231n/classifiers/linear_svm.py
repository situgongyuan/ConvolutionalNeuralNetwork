import numpy as np

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  n_sample = X.shape[0]
  hinge_loss = 0.0
  for i in xrange(n_sample):
    scores = np.dot(X[i],W)
    correct_class_score = scores[y[i]]
    margin = np.maximum(0,scores - correct_class_score + 1)
    margin[y[i]] = 0
    hinge_loss += np.sum(margin) / n_sample

    margin = scores - correct_class_score + 1
    margin = (margin > 0) * 1
    margin[y[i]] = 0
    margin[y[i]] = -np.sum(margin)
    dW_single_sample = np.outer(X[i],margin) / n_sample
    dW += dW_single_sample + reg * W / n_sample
    '''
    for j in xrange(n_class):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        data_loss += margin / n_sample
    '''
  reg_loss = 0.5 * reg * np.sum(W ** 2)
  # Add regularization to the loss.
  loss = hinge_loss + reg_loss

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  n_sample, n_feature  = X.shape
  assert n_feature == W.shape[0]

  scores = np.dot(X,W)
  correct_class_scores = scores[np.arange(n_sample),y].reshape(n_sample,1)
  scores = np.maximum(0, scores - correct_class_scores + 1.0)

  scores[np.arange(n_sample),y] = 0
  num_pos = np.sum(scores > 0, axis = 1)

  hinge_loss = np.sum(scores) / n_sample
  reg_loss = 0.5 * reg * np.sum(W ** 2)
  loss = hinge_loss + reg_loss

  dscores = np.zeros(scores.shape)
  dscores[scores > 0] = 1
  dscores[np.arange(n_sample),y] -= num_pos
  dW = np.dot(X.T,dscores) / n_sample + reg * W

  return loss, dW

if __name__ == "__main__":
  x = np.random.randn(5,6)
  y = np.asarray([0,1,0,2,1])
  W = np.random.rand(6,3) + 2
  reg = 0.001
  print svm_loss_naive(W,x,y,reg)
  print svm_loss_vectorized(W,x,y,reg)

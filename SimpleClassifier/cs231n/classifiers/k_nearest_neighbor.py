import numpy as np
from collections import Counter


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i,j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis = 1))
    return dists


  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    """
    (num_train,num_feature) = self.X_train.shape
    transformed_train_x = self.X_train.reshape(num_train,1,num_feature)
    dists = np.sqrt(np.sum((transformed_train_x - X) ** 2, axis = 2)).T
    return dists
    """
    Y = self.X_train
    XY = np.dot(X,Y.T)
    XX = np.sum(X ** 2,axis = 1,keepdims = True)
    YY = np.sum(Y ** 2,axis = 1)
    dists = np.sqrt(XX - 2 * XY + YY)
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      sorted_k_index = np.argsort(dists[i])[:k]
      k_nearest_neighbours_labels = self.y_train[sorted_k_index]
      y_pred[i] = Counter(k_nearest_neighbours_labels).most_common(1)[0][0]
    return y_pred

if __name__ == "__main__":
  x = np.random.rand(5,6)
  y = np.arange(5)
  knn = KNearestNeighbor()
  knn.train(x,y)
  test_x = np.random.rand(2,6)
  print knn.compute_distances_no_loops(test_x)
  print knn.compute_distances_one_loop(test_x)
  print knn.compute_distances_two_loops(test_x)
  dists = knn.compute_distances_no_loops(test_x)
  print knn.predict_labels(dists, k = 2)

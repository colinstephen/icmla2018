#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

'''
About
=====

Defines and implements `NearestKDEClassifier()`: a Sklearn-compatible class that
learns per-class sub level set persistence diagram weighted KDEs [1,2] for
arrays of time series, and predicts new labels using the minimum Sinkhorn
divergence [3] between KDEs. The associated function `sinkhorn_distance_2d()`
computes the Sinkhorn divergence for 2d histograms/KDEs over Lp based cost
matrices using structured matrix vector product optimizations.

Implements time series classifier benchmarks `DTWClassifier()`,
`CepstralClassifier()`, and a `TopologicalEntropyTransformer()` for use in
sklearn pipelines [4,5,6].

Implements an optimised version of Sinkhorn divergence [3] computed between 2d
distributions using L_p based cost matrices. Enhancements include FFT based
convolutions and leveraging the block matrix structure of the resulting kernels.

References
==========

[1] Adams et. al., Persistence images: A stable vector representation of
persistent homology. The Journal of Machine Learning Research 18, 1 (2017),
218–252.

[2] Chazal et. al., The density of expected persistence diagrams and its kernel
based estimation. ArXiv e-prints (Feb. 2018).

[3] Cuturi, Sinkhorn distances: Lightspeed computation of optimal transport. In
Advances in neural information processing systems (2013), pp. 2292–2300.

[4] Bagnall et. al., The great time series classification bake off: a review and
experimental evaluation of recent algorithmic advances. Data Mining and
Knowledge Discovery 31, 3 (May 2017), 606–660.

[5] Kalpakis et. al., Distance measures for effective clustering of arima
time-series. In Proceedings 2001 IEEE International Conference on Data Mining
(2001), pp. 273–280.

[6] Rucco et. al., A new topological entropy-based approach for measuring
similarities among piecewise linear functions. Signal Processing 134 (2017),
130–138.

'''


import sys
import subprocess
import numpy as np
import scipy as sp
from functools import wraps
from functools import partial
from tempfile import NamedTemporaryFile
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from itertools import product
from scipy.signal import convolve
from scipy.signal import gaussian
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic_2d
from scipy.linalg import toeplitz
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from dtaidistance import dtw
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()


###############
# CLASSIFIERS #
###############


class NearestKDEClassifier(BaseEstimator, ClassifierMixin):
  '''
  Classify time series using nearest class KDE under the Sinkhorn metric.
  ''' 
  def __init__(self, grid_size=25, weight_func='uniform', weight_params=None,
    kernel_std=0.5, kernel_radius=5, sinkhorn_reg=1, sinkhorn_method='vectrick',
    lp_norm=1, embed=False, embed_dim=None, embed_lag=None, maxdimension=1,
    maxscale=None, n_jobs=1):
    '''
    Parameters:
        * grid_size - number of boxes along each dimension
        * weight_func - the weight function applied to points in a (b,p) PD
        * weight_params - dictionary of named parameters for weight func
        * kernel_std - standard deviation of the spherical Gaussian kernel
        * kernel_radius - extent of kernel in each direction (nb of stds)
        * sinkhorn_reg - Sinkhorn regularization parameter
        * sinkhorn_method - choice of algorithm for matrix-vector products
        * lp_norm - underlying p-value for Sinkhorn cost matrix M
        # embed - whether to embed the time series before computing PD
        * embed_dim - dimension of Takens embedding if required
        * embed_lag - lag of Takens embedding if required
        * persistence_dimension - which dimension of features to consider
        * maxscale - stop the filtration at this value
        # n_jobs - how many processes to use

    TODO:
        * When embed=True and maxdimension=0 we don't need to use a full 2d
        grid, since the Rips filtration barcodes all start at value 0. In
        this situation we can compute Sinkhorn divergence directly between
        two 1d distributions. 
    '''
    self.grid_size = grid_size
    self.weight_func = weight_func
    self.weight_params = weight_params
    self.kernel_std = kernel_std
    self.kernel_radius = kernel_radius
    self.sinkhorn_reg = sinkhorn_reg
    self.sinkhorn_method = sinkhorn_method
    self.lp_norm = lp_norm
    self.embed = embed
    self.embed_dim = embed_dim
    self.embed_lag = embed_lag
    self.maxdimension = maxdimension
    self.maxscale = maxscale
    self.n_jobs = n_jobs

  def fit(self, X, y=None):

    if y is None:
      raise ValueError("Classifier requires target labels to fit.")

    if self.weight_func == 'linear':
      self.weight_func_ = linear_weight
    elif self.weight_func == 'normal':
      self.weight_func_ = normal_weight
    elif self.weight_func == 'uniform':
      self.weight_func_ = uniform_weight
    elif self.weight_func == 'sigmoid':
      self.weight_func_ = sigmoid_weight
    elif self.weight_func == 'ramp':
      self.weight_func_ = ramp_weight
    else:
      raise ValueError("Invalid weight function {}".format(self.weight_func))

    if self.weight_params is not None:
      self.weight_func_ = partial(self.weight_func_, **self.weight_params)

    if self.embed and self.maxdimension == 0:
      self._hist_func = pd_weighted_hist_1d
      self._kde_func = pd_kde_1d
      self._sinkhorn_func = sinkhorn_distance_1d
    else:
      self._hist_func = pd_weighted_hist
      self._kde_func = pd_kde
      self._sinkhorn_func = sinkhorn_distance_2d

    # estimate lag as the mean of lag estimates
    if self.embed and self.embed_lag is None:
      lags = [estimate_lag(x) for x in X]
      self.embed_lag = int(np.round(np.mean(lags)))
      print('Estimated embedding lag as', self.embed_lag)

    # estimate dimension as the mean of dimension estimates
    if self.embed and self.embed_dim is None:
      dims = [estimate_dim(x, self.embed_lag) for x in X]
      self.embed_dim = int(np.round(np.mean(dims)))
      print('Estimated embedding dim as', self.embed_dim)

    # estimate max scale for filtration
    if self.embed and self.maxscale is None:
      # TODO: a sensible implementation
      pass

    # Treat each class as a separate group
    self.classes_ = np.unique(y)
    X_ = [X[y==l] for l in self.classes_]
    
    # Get the persistence diagrams for each class
    if self.embed:
      X_ = [Parallel(n_jobs=self.n_jobs)(delayed(ripser_diag_bd)(x,
        self.embed_dim, self.embed_lag, self.maxdimension, self.maxscale)
        for x in X) for X in X_]
    else:
      X_ = [[diag_bd(x) for x in X] for X in X_]

    # Convert to a class representation of the birth-persistence pairs
    X_ = [[diag_bp(x) for x in X] for X in X_]
    X_ = [np.concatenate(X) for X in X_]

    # Scale the data using the overall min and max values in each dimension
    self.mins_ = np.min([np.min(X, axis=0) for X in X_], axis=0)
    self.maxs_ = np.max([np.max(X, axis=0) for X in X_], axis=0)
    X_ = [(X - self.mins_) / (self.maxs_ - self.mins_) for X in X_]

    # Record the fitted diagrams for visualization
    self.fit_pds_ = X_.copy()

    # Extend a grid slightly beyond the (0,1) scaling for test data
    self.x_edges = np.linspace(-0.25, 1.25, self.grid_size + 1)
    self.y_edges = np.linspace(0.0, 1.5, self.grid_size + 1)

    # Bin points to compute the histogram representation
    X_ = [self._hist_func(X, self.x_edges, self.y_edges,
      weight_func=self.weight_func_, normed=True) for X in X_]

    # Record the weighted histograms for visualization
    self.fit_class_hists_ = X_.copy()

    # Convolve with a Gaussian
    X_ = [self._kde_func(X, std=self.kernel_std, radius=self.kernel_radius,
      normed=True) for X in X_]

    # Record the weighted KDEs for visualization
    self.fit_class_kdes_ = X_.copy()

    # Store the fitted data as an array
    self.X_ = np.array(X_)

    return self

  def transform(self, X, y=None):
      
    # Find (birth, persistence) diagram representations of each row
    if self.embed:
      X_ = Parallel(n_jobs=self.n_jobs)(delayed(ripser_diag_bd)(x,
        self.embed_dim, self.embed_lag, self.maxdimension, self.maxscale)
        for x in X)
    else:
      X_ = [diag_bd(x) for x in X]

    X_ = [diag_bp(x) for x in X_]

    # Scale the data using the fitted min and max values in each dimension
    X_ = [(X - self.mins_) / (self.maxs_ - self.mins_) for X in X_]

    # Clip values outside the grid limits
    X_ = [np.clip(X, [-0.25, 0.0], [1.25, 1.5]) for X in X_]

    # Bin points to compute the histogram representation
    X_ = [self._hist_func(X, self.x_edges, self.y_edges,
      weight_func=self.weight_func_, normed=True) for X in X_]

    # Convolve with a Gaussian
    X_ = [self._kde_func(X, std=self.kernel_std, radius=self.kernel_radius,
      normed=True) for X in X_]
    
    # Return the data as an array
    X_ = np.array(X_)
    
    return X_       

  def predict(self, X, y=None):

    X_ = self.transform(X)

    # Compute pairwise distances between the test and fit representations 
    dists = Parallel(n_jobs=self.n_jobs)(delayed(self._sinkhorn_func)(Xi,
      Xj, reg=self.sinkhorn_reg, lp_norm=self.lp_norm,
      method=self.sinkhorn_method) for Xi in X_ for Xj in self.X_)

    # Determine the closest class indexes
    m, n = len(X_), len(self.classes_)
    dists = np.array(dists).reshape((m,n)) 
    idx = np.argmin(dists, axis=1)
    
    # The corresponding labels are our predictions
    return self.classes_[idx]


class DTWClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classify time series using one nearest neighbour under the
    classic Dynamic Time Warp distance using a fractional window.
    '''

    def __init__(self, window_size=None, n_jobs=1):
        self.window_size = window_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # No fitting required!
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        return self

    def predict(self, X, y=None):
        m = len(X)
        n = len(self.X_)
        if self.window_size is None:
          self.window_size = 100
        dists = Parallel(n_jobs=self.n_jobs)(delayed(dtw.distance_fast)(xi, xj,
            window=self.window_size) for xi in X for xj in self.X_)
        dists = np.array(dists).reshape((m,n))
        idx = np.argmin(dists, axis=1)
        return self.y_[idx]


class CepstralClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classify time series using one nearest neighbour under Euclidean distance
    between cepstral coefficients.

    Randall, Robert B. "A history of cepstrum analysis and its application to
    mechanical problems." Mechanical Systems and Signal Processing 97 (2017):
    3-19.
    '''

    def __init__(self, num_coeffs=None, n_jobs=1):
        self.num_coeffs = num_coeffs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # No fitting required!
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        return self

    def predict(self, X, y=None):
        m = len(X)
        n = len(self.X_)
        result = Parallel(n_jobs=self.n_jobs)(delayed(cepstral_distance)(xi, xj,
          self.num_coeffs) for xi in X for xj in self.X_)
        dists = np.array(result).reshape((m,n))
        idx = np.argmin(dists, axis=1)
        return self.y_[idx]


def cepstral_coeffs(signal, num_coeffs=None):
    powerspectrum = np.abs(np.fft.fft(signal))**2
    powerspectrum += 1e-32
    cepstrum = np.abs(np.fft.ifft(np.log(powerspectrum)))**2
    coeffs = sp.fftpack.dct(cepstrum)
    return np.real(coeffs[:num_coeffs])


def cepstral_distance(v, w, num_coeffs=None):
    c1 = cepstral_coeffs(v, num_coeffs=num_coeffs)
    c2 = cepstral_coeffs(w, num_coeffs=num_coeffs)
    clip = min(len(c1), len(c2))
    return np.linalg.norm(c1[:clip] - c2[:clip])


class TopologicalEntropyTransformer(BaseEstimator, TransformerMixin):
    '''
    Compute the persistent entropy of each time series passed in.
    
    Rucco, Matteo, et al. "A new topological entropy-based approach for
    measuring similarities among piecewise linear functions." Signal Processing
    134 (2017): 130-138.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pds = [diag_bd(x) for x in X]
        entropies = [persistent_entropy(pd) for pd in pds]
        return np.array(entropies).reshape(-1,1)
        

def persistent_entropy(pd):
    xi, yi = pd[:,0], pd[:,1]
    li = yi - xi
    L = np.sum(li)
    pi = li / L
    H = -np.sum(pi * np.log(pi))
    return H


#######################
# SINKHORN DIVERGENCE #
#######################


def sinkhorn_distance_1d(*args, **kwargs):
    raise NotImplementedError('TODO')


def sinkhorn_distance_2d(a, b, reg=0.25, lp_norm=1, numItermax=1000,
                         stopThr=1e-6, method='vectrick', verbose=False):
  '''
  Compute Sinkhorn distance (regularized optimal transport) between 2d
  histograms / discrete KDEs using convolutions or the vec trick. The
  underlying cost matrix is Lp for the given value of lp_norm.

  Note that the choice of method dictates which representation should be used
  for the arrays a and b (see below).

  See http://marcocuturi.net/Code/sinkhornTransport.m for Cuturi's original
  Matlab implementation that does not use the matrix-vector product speedups
  implemented here.

  TODOs:

    * Implement 1-vs-N and N-vs-N versions to compare multiple distributions
    simultaneously.
    * Return lower bounds to the classical OT distance (Sinkhorn
    distance/divergence is an upper bound) using the dual variables from the
    optimization.
  
  Parameters:

    * a - square matrix representing histogram or KDE
    * b - square matrix representing histogram or KDE with same shape as a
    * reg - Sinkhorn regularization paramter
    * lp_norm - underlying p-value for Sinkhorn cost matrix M
    * numItermax - max iterations of Sinkhorn-Knopp fixed point algorithm
    * stopThr - bound on relative change of SK estimates for convergence
    * method - 'vectrick' or 'convolution' matrix-vector product algorithm
    * verbose - print relative change data as SK algorithm executes

  Returns:

      * Sinkhorn divergence (upper bound to OT distance)
  '''

  # Validate the matrix-vector product method.
  valid_methods = ['vectrick', 'convolution']
  if method not in valid_methods:
    msg = "Sinkhorn method should be one of {}.".format(valid_methods)
    raise ValueError(msg)

  # Validate that the input distributions are compatible.
  if a.shape != b.shape:
    msg = "Input distribution shapes must match. Got {} and {}."
    msg = msg.format(a.shape, b.shape)
    raise ValueError(msg)

  # Get the grid size
  n = a.shape[0]

  # Set up cost and kernel matrix generators.
  # Depending on the method, different representations are required.
  # Underscore represents that we work with generators and not the matrices.
  if method == 'convolution':
    M_ = M_conv(n, p=lp_norm)
  elif method == 'vectrick':
    M_ = M_block(n, p=lp_norm)
  K_ = K_block(M_, reg)

  # Use vectors from now on.
  a = np.ravel(a)
  b = np.ravel(b)

  # Initial projections correspond to the uniform distribution.
  u = np.ones_like(a) / len(a) # + 1e-32
  v = np.ones_like(b) / len(b) # + 1e-32

  # Reserve memory for matrix-vector products K@u and K@v.
  Kx = np.empty_like(a)
  
  # Also keep full copies of u, v in case the algorithm explodes.
  uprev = np.empty_like(u)
  vprev = np.empty_like(v)

  # initialize loop variables
  # Dprev is the previous estimate of Sinkhorn divergence
  loop = 0
  D = Dprev = 1

  # loop to compute the iterated projections
  while (loop < numItermax):
      
    # copy the last good values
    uprev[:] = u
    vprev[:] = v

    # update v
    if method == 'convolution':
      Kx[:] = BTTB_mult(K_, u)
    elif method == 'vectrick':
      Kx[:] = K_times_x(K_, u)
    if np.any(Kx == 0):
      print('Machine precision reached at iteration', loop)
      u = uprev
      v = vprev
      break
    v[:] = b / Kx

    # update u
    if method == 'convolution':
      Kx[:] = BTTB_mult(K_, v)
    elif method == 'vectrick':
      Kx[:] = K_times_x(K_, v)
    if np.any(Kx == 0):
      print('Machine precision reached at iteration', loop)
      u = uprev
      v = vprev
      break
    u[:] = a / Kx

    # ensure current vectors are usable
    if (np.any(np.isnan(u)) or np.any(np.isinf(u)) or np.any(np.isnan(v)) or
      np.any(np.isinf(v))):
      
      print('Machine precision reached at iteration', loop)
      u = uprev
      v = vprev
      break

    # recalculate the relative decrease in final transport cost D
    if loop % 10 == 0 or loop + 1 == numItermax:

      if method == 'convolution':
        Kx[:] = BTTB_mult(M_*K_, v)
      elif method == 'vectrick':
        Kx[:] = MK_times_x(M_, K_, v)

      D = np.dot(u, Kx)
      err = np.abs(D/Dprev - 1)

      if err < stopThr or np.isnan(err):
        break
      else:
        Dprev = D

      if verbose:
        if loop % 100 == 0:
          print_log_header()
        print_log(loop, err, D)

    loop += 1

  return D


#####################
# UTILITY FUNCTIONS #
#####################

'''
Useful conversions, estimators of various parameters, and importantly fast
matrix-vector product optimisations that can be used with the structured block
matrices (cost matrix and kernel matrix) used in computing Sinkhorn divergence.

Several methods simply wrap an existing implementation in R where it exists,
however the matrix operations are implemented directly in Numpy code.
'''


# R function to compute sublevel set persistence diagrams of time series.
# TODO: use GUDHI Python API rather than calling R.
robjects.r('''
  pd = function(time_series) {
    N = length(time_series)
    FUN = function(data, grid) { rep(data, times = 2) }
    lim = c(0, N-1)
    by = 1
    TDA::gridDiag(
      time_series,
      FUN = FUN,
      lim = lim,
      by = by,
      location = FALSE,
      maxdimension = 0,
      library = 'PHAT')
  }
  ''')

R_pd = robjects.r['pd']
R_TDA = importr('TDA')
R_NLTS = importr('nonlinearTseries')


def atleast_2d(f):
  # Decorator to ensure wrapped function works with n>=0 points.
  @wraps(f)
  def wrapper(X, *args, **kwargs):
    X = np.atleast_2d(X)
    return f(X, *args, **kwargs)
  return wrapper


def delay_coords(x, dim=2, lag=1):
  # Compute the embedding coordinates of a time series embedding
  dim = int(np.round(dim))
  lag = int(np.round(lag))
  X = np.empty((dim, len(x)))
  for d in range(dim):
    X[d,:] = np.roll(x, -d*lag)
  return X[:,:-(dim-1)*lag].T


def estimate_dim(x, lag):
  # Given a time series estimate the embedding dim for a Takens embedding
  lag = int(np.round(lag))
  dim = R_NLTS.estimateEmbeddingDim(x, time_lag=lag, max_embedding_dim=10,
    do_plot=False)
  return int(dim[0])


def estimate_lag(x):
  # Given a time series estimate the lag for a Takens embedding
  lag_max=len(x)//5
  try:
    lag = R_NLTS.timeLag(x, technique='ami', selection_method='first.minimum',
      lag_max=lag_max, do_plot=False)
    return int(lag[0])
  except:
    # there may be no minimum
    return lag_max
    

def rips_diag_bd(x, dim, lag, maxdimension, maxscale):
  # Get the rips filtration persistence diagram for an embedded time series
  maxdimension_ = max(maxdimension, 1)
  X = delay_coords(x, dim, lag)
  R_diagram = R_TDA.ripsDiag(X, maxdimension_, maxscale)
  diagram = np.array(R_diagram.rx2('diagram'))
  idx = diagram[:,0] == maxdimension
  return diagram[idx,1:]


def ripser_diag_bd(x, dim, lag, maxdimension, maxscale):
  # As with rips_diag_bd but use ripser rather than R TDA to compute the PD
  X = delay_coords(x, dim, lag)
  with NamedTemporaryFile('w') as f:
    np.savetxt(f.name, X)
    result = subprocess.check_output(['ripser', '--format', 'point-cloud',
      '--dim', str(maxdimension), '--threshold', str(maxscale), f.name])
  lines = result.decode().split('\n')
  divider_text = 'persistence intervals in dim {}:'
  start_line = lines.index(divider_text.format(maxdimension)) + 1
  try:
    end_line = lines.index(divider_text.format(maxdimension+1))
  except ValueError:
    end_line = -1
  interval_lines = lines[start_line:end_line]
  intervals = [line[2:-1].split(',') for line in interval_lines]
  intervals = [[b,d] if d != ' ' else [b,maxscale] for b,d in intervals]
  return np.array(intervals, dtype=np.double)


def diag_bd(x):
  # Get the sub level set persistence diagram of a time series
  R_diagram = R_pd(np.array(x))
  return np.array(R_diagram.rx2('diagram'))[:,1:]


@atleast_2d
def diag_bp(x):
  # Change persitence diagram representation to (birth, persistence)
  return np.vstack((x[:,0], x[:,1]-x[:,0])).T


@atleast_2d
def weights(pd, weight_func=None):
  # Compute the weight of each point in a persistence diagram
  if weight_func is None:
    weight_func = uniform_weight
  return weight_func(pd)


@atleast_2d
def sigmoid_weight(X, midpoint=0.0, slope=20):
  # Apply a sigmoid function to each persistence value to compute its weight.
  return (1 / (1 + np.exp(-slope * (X[:,1]-midpoint)))) - (1 / (1 + np.exp(
    -slope * (-midpoint))))


@atleast_2d
def linear_weight(X):
  # Each point weight is just the persistence value.
  return np.where(X[:,1] < 1, X[:,1], 1)


@atleast_2d
def uniform_weight(X):
  # Each point weight is one.
  return np.where(X[:,1] > 0.001, 1, 0)


@atleast_2d
def normal_weight(X, std=0.2, mean=0.5):
  # Apply normal distribution over persistence values
  norm = np.exp((-(X[:,1]-mean)**2)/(2*std**2)) - np.exp((-(-mean)**2)/(2 * std
    ** 2))
  return np.where(norm > 0, norm, 0)


@atleast_2d
def ramp_weight(X, thresh=0.3):
  return np.where(X[:,1] < thresh, X[:,1]/thresh, 1)


@atleast_2d
def pd_weighted_hist_1d(*args, **kwargs):
  raise NotImplementedError('TODO')


@atleast_2d
def pd_weighted_hist(pd, x_edges, y_edges, weight_func=None, normed=True):
  # Quantize a persistence diagram over a regular grid.
  b = pd[:,0]
  p = pd[:,1]
  bins = x_edges, y_edges
  w = weights(pd, weight_func=weight_func)
  hist, _1, _2, _3 = binned_statistic_2d(b, p, w, statistic='sum', bins=bins)
  np.nan_to_num(hist, copy=False)
  if normed:
      hist /= np.sum(hist)
  return hist


def pd_kde_1d(*args, **kwargs):
  raise NotImplementedError('TODO')


def pd_kde(hist, std=0.1, radius=5, normed=True):
  '''
  Given a histogram convolve it with a Gaussian.

  Uses a spherically symmetric discrete Gaussian filter that extends
  to 5 standard deviations on each side by default.

  The `scipy.signal.convolve()` method used here will use FFT based convolution
  when that is predicted to be the fastest method.
  '''
  window = gaussian(1 + (std * radius * 2 // 1), std=std)
  kernel = np.outer(window, window)
  I = convolve(hist, kernel, mode='same')
  return I / np.sum(I) if normed else I


def BTTB_mult(t, v):
  '''
  Speed up Block Toeplitz of Toeplitz Blocks matrix-vector multiplications using
  convolution products.

  Returns T@v where T is the Block Toeplitz of Toeplitz Blocks matrix generated
  by t. In particular this can be used to compute K@v for kernels K arising from
  Lp cost matrices:

    { M' = M_conv(n) & K' = K_block(M') } =>

    { K @ v = vec(convolve(K', arr(v))) }

  See Algorithm 5.2.1 in Chapter 5 of of Vogel (2002) "Computational Methods for
  Inverse Problems", for why this works. In brief block Toeplitz of Toeplitz
  block matrices can be embedded in block circulant of circulant blocks
  matrices, the latter of which have periodic structure. The periodic structure
  can be exploited by replacing matrix algebra operations with simpler
  operations in the Fourier basis.

  Note that the scipy method `signal.convolve()` used here implements either
  direct or FFT convolutions depending on which will be faster.
  '''
  return vec(convolve(t, arr(v), mode='valid'))


def print_log_header():
  '''
  Prepare to print loop number and relative error table
  '''
  print('{:5s}|{:12s}|{:12s}'.format('It.', 'Err', 'Dist') + '\n' + '-' * 19)


def print_log(loop, err, dist):
  '''
  Print loop number and current relative error
  '''
  print('{:5d}|{:8e}|{:8e}'.format(loop, err, dist))


def vec(V):
  '''
  Stack array V as a vector, using columns as contiguous blocks.
  '''
  return V.ravel(order='F')


def arr(v):
  '''
  Unstack vector v as a square matrix V, with each column of V being a
  contiguous block of v.
  '''
  n = np.int(np.sqrt(len(v)))
  return v.reshape(n, n, order='F')


def M_conv(n, p=1):
  '''    
  Create a generator array M' for a n^2-by-n^2 block Toeplitz of Toeplitz blocks
  matrix M representing Lp distances (to power p) on an n-by-n grid. 

  Generation of M from M' is via the method outlined in Proposition 5.26 (p.72)
  of Vogel (2002) "Computational Methods for Inverse Problems", SIAM.
  '''
  deltas = np.abs(np.linspace(1 - n, n - 1, num=2 * n - 1))

  # reshape to allow broadcasting
  xx = deltas.reshape(-1,1)
  yy = deltas.reshape(1,-1)

  # compute the grid distances for each block
  if p == np.inf:
    gen = np.maximum(xx, yy)
  else:
    gen = np.abs(xx)**p + np.abs(yy)**p
  
  return gen


def M_block(n, p=1):
  '''
  Return generator matrix M' for a n^2-by-n^2 block Toeplitz of Toeplitz blocks
  matrix M representing Lp distances (to power p) on an n-by-n grid. 

  Generation of M is via symmetric tensor product with the matrix of ones and
  the matrix returned from this function:

    { M' = M_block(n, p) => M = (M'⊗1) + (1⊗M') }
  '''
  return toeplitz(np.arange(n)**p)


def K_block(M_gen, reg=0.5):
    '''
    Given a generator M_gen of a cost matrix M for Lp distances on an n-by-n
    grid, return a generator for the associated kernel matrix K used in the
    Sinkhorn Knopp algorithm.
    '''
    return np.exp(-M_gen/reg)


def tensor(A, B=None):
  ''' 
  Utility function for testing.
  '''
  if B is None:
    B = A
  return np.kron(A,B)


def symmetric_tensor(A, B=None):
  '''
  Utility function for testing.
  '''
  if B is None:
    B = np.ones_like(A)
  return tensor(A,B) + tensor(B,A)


def vec_trick(A, B, v):
  '''
  Vec trick computes the result of (A⊗B)v using only A and B matrix products.
  '''
  return vec(B @ arr(v) @ A.T)


def K_times_x(K_, x):
  '''
  Speed up kernel matrix-vector multiplications using the block representation
  of K.

    { M' = M_block(n, p) => M = (M'⊗1) + (1⊗M') } =>

    { K' = K_block(M', reg) => K = (K'⊗K') } =>

    { K @ x = (K'⊗K') @ x }
  '''
  return vec_trick(K_, K_, x)


def MK_times_x(M_, K_, x):
  '''
  Speed up (cost * kernel) matrix-vector multiplications using the block
  representations of M and K.

    { M' = M_block(n, p) => M = (M'⊗1) + (1⊗M') } =>

    { K' = K_block(M', reg) => K = (K'⊗K') } =>

    { (M * K) @ x = ((M' * K') ⊗ K' + K' ⊗ (M' * K')) @ x }
  '''
  MK_ = M_ * K_
  return vec_trick(MK_, K_, x) + vec_trick(K_, MK_, x)


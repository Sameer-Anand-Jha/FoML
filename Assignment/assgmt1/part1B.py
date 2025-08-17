import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2

def solve(N, d):
  U, M1, M2 = initialise_input(N, d)

  '''
  Enter your code here for steps 1 to 6
  '''
  return max_indices
  
solve(N,d)

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

  '''
  Enter your code here for steps 1 to 6
  '''
  U, M1, M2 = initialise_input(N, d)
  
  # Q.1
  X = U.dot(M1)
  Y = U.dot(M2)

  # print(f"X:{X}\nY:{Y}\n")

  # Q. 2
  I= np.arange(1, N+1).reshape(N, 1)
  X_hat = X + I
  Y_hat = Y + I

  # print(f"X_hat:{X_hat}\nY_hat:{Y_hat}")

  # Q.3
  Z = X_hat.dot(Y_hat.T)
  # print(f"Z:{Z}\n")
  
  sparser1 = np.resize([1, 0], N)
  mask1 = np.arange(N) % 2 == 0
  sparser2 = np.resize([0, 1], N)
  mask2 = np.arange(N) % 2 == 1

  Z[mask1] *= sparser1
  Z[mask2] *= sparser2

  # print(f"sparsed Z:{Z}\n")

  # Q.4
  Z_hat = np.exp(Z)
  Z_hat = Z_hat/Z_hat.sum(axis=1)[:, None]

  # print(f"row_sums - Z:{Z_hat.sum(axis=1)}")

  # Q.5
  max_indices = Z_hat.argmax(axis=1)
  print(f"max indices : {max_indices}")

  return max_indices
  
solve(5, 3)
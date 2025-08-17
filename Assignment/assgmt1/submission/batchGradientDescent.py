import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

class LinearRegressionBatchGD:
  def __init__(self, learning_rate=0.01, max_epochs=100, batch_size=10):
    '''
    Initializing the parameters of the model

    Args:
      learning_rate : learning rate for batch gradient descent
      max_epochs : maximum number of epochs that the batch gradient descent algorithm will run for
      batch-size : size of the batches used for batch gradient descent.

    Returns:
      None
    '''
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.weights = None

  def fit(self, X, y, plot=True):
    '''
    This function is used to train the model using batch gradient descent.

    Args:
      X : 2D numpy array of training set data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)

    Returns :
      None
    '''
    if self.batch_size is None:
      self.batch_size = X.shape[0]

    # Initialize the weights
    self.weights = np.zeros((X.shape[1],1))

    prev_weights = self.weights

    self.error_list = []  #stores the loss for every epoch
    for epoch in range(self.max_epochs):

      batches = create_batches(X, y, self.batch_size)
      for batch in batches:
        X_batch, y_batch = batch  #X_batch and y_batch are data points and target values for a given batch

        # Complete the inner "for" loop to calculate the gradient of loss w.r.t weights, i.e. dw and update the weights
        # You should use "compute_gradient()"  function to calculate gradient.
        prev_weights = self.weights
        self.weights = self.weights - self.learning_rate * self.compute_gradient(X_batch, y_batch, prev_weights)

      # After the inner "for" loop ends, calculate loss on the entire data using "compute_rmse_loss()" function and add the loss of each epoch to the "error list"
      self.error_list.append(self.compute_rmse_loss(X, y, self.weights))

      if np.linalg.norm(self.weights - prev_weights) < 1e-5:
        # print(f" Stopping at epoch {epoch}.")
        break
    
    if plot:
        plot_loss(self.error_list, epoch+1)

  def predict(self, X):
    '''
    This function is used to predict the target values for the given set of feature values

    Args:
      X: 2D numpy array of data points. Dimensions (n x (d+1))

    Returns:
      2D numpy array of predicted target values. Dimensions (n x 1)
    '''
    # Write your code here
    return X.dot(self.weights)

  def compute_rmse_loss(self, X, y, weights):
    '''
    This function computes the Root Mean Square Error (RMSE)

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)
      weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)

    Returns:
      loss : 2D numpy array of RMSE loss. float
    '''
    # Write your code here
    error = y - X.dot(weights)
    return np.sqrt(np.sum(error.T.dot(error))/X.shape[0])

  def compute_gradient(self, X, y, weights):
    '''
    This function computes the gradient of mean squared-error loss w.r.t the weights

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)
      weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)

    Returns:
      dw : 2D numpy array of gradients w.r.t weights. Dimensions ((d+1) x 1)
    '''
    # Write your code here.
    # Note: Make sure you divide the gradient (dw) by the total number of training instances before returning to prevent "exploding gradients".
    return (X.T.dot(X.dot(weights) - y) )/ X.shape[0]

def plot_loss(error_list, total_epochs):
  '''
  This function plots the loss for each epoch.

  Args:
    error_list : list of validation loss for each epoch
    total_epochs : Total number of epochs
  Returns:
    None
  '''
  # Complete this function to plot the graph of losses stored in model's "error_list"
  # raise NotImplementedError()
  plt.plot(np.arange(0, total_epochs, 1), error_list)
  plt.title("Plot for the equation of the form: y = w0 + w1*x")
  plt.xlabel("Epochs")
  plt.ylabel("RMSE loss")
  plt.savefig("plot_loss.png")
  plt.close()

def plot_learned_equation(X, y, y_hat):
    '''
    This function generates the plot to visualize how well the learned linear equation fits the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 2)
      y : 2D numpy array of target values. Dimensions (n x 1)
      y_hat : 2D numpy array of predicted values. Dimensions (n x 1)

    Returns:
      None
    '''
    # Plot a 2d plot, with only  X[:,1] on x-axis (Think on why you can ignore X[:, 0])
    # Use y_hat to plot the line. DO NOT use y.
    plt.scatter(X[:, 1], y.squeeze())
    plt.plot(X[:, 1], y_hat, c="red")
    plt.title("Plot for the equation of the form: y = w0 + w1*x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("gradient_descent.png")
    plt.close()

############################################
#####        Helper functions          #####
############################################
def generate_toy_dataset():
    '''
    This function generates a simple toy dataset containing 300 points with 1d feature 
    '''
    X = np.random.rand(300, 2)
    X[:, 0] = 1 # bias term
    weights = np.random.rand(2,1)
    noise = np.random.rand(300,1) / 32
    y = np.matmul(X, weights) + noise
    
    X_train = X[:250]
    X_test = X[250:]
    y_train = y[:250]
    y_test = y[250:]
    
    return X_train, y_train, X_test, y_test

def create_batches(X, y, batch_size):
  '''
  This function is used to create the batches of randomly selected data points.

  Args:
    X : 2D numpy array of data points. Dimensions (n x (d+1))
    y : 2D numpy array of target values. Dimensions (n x 1)

  Returns:
    batches : list of tuples with each tuple of size batch size.
  '''
  batches = []
  data = np.hstack((X, y))
  np.random.shuffle(data)
  num_batches = data.shape[0]//batch_size
  i = 0
  for i in range(num_batches+1):
    if i<num_batches:
      batch = data[i * batch_size:(i + 1)*batch_size, :]
      X_batch = batch[:, :-1]
      Y_batch = batch[:, -1].reshape((-1, 1))
      batches.append((X_batch, Y_batch))
    if data.shape[0] % batch_size != 0 and i==num_batches:
      batch = data[i * batch_size:data.shape[0]]
      X_batch = batch[:, :-1]
      Y_batch = batch[:, -1].reshape((-1, 1))
      batches.append((X_batch, Y_batch))
  return batches


# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Gradient descent solution for linear regression #####")
    
    # Hyperparameters
    learning_rate = 0.01
    batch_size = 5 # None
    max_epochs = 100
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_test, y_test = generate_toy_dataset()
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    print(RESET + "Calculating closed form solution: ", end="")
    try:
        linear_reg = LinearRegressionBatchGD(learning_rate=learning_rate, max_epochs=max_epochs, batch_size=5)
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Predicting for test split: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Plotting the solution: ", end="")
    try:
        plot_learned_equation(X_test, y_test, y_hat)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
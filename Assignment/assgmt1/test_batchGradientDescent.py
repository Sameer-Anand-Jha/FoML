import pytest
from batchGradientDescent import LinearRegressionBatchGD
import numpy as np

def test_obj_init():
    obj = LinearRegressionBatchGD()

def test_fit_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    obj = LinearRegressionBatchGD()
    obj.fit(X, y, plot=False)

def test_predict_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    obj = LinearRegressionBatchGD()
    obj.fit(X, y, plot=False)
    
    y_hat = obj.predict(X)
    assert y_hat.shape == y.shape, "Shapes don't match for predict() function"
    
def test_rmse_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    weight = np.random.rand(2,1)
    obj = LinearRegressionBatchGD()
    obj.fit(X, y, plot=False)
    
    loss = obj.compute_rmse_loss(X, y, weight)
    assert isinstance(loss, float), "Shapes don't match for compute_rmse_loss() function"
    
def test_computing_gradient_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    weight = np.random.rand(2,1)
    obj = LinearRegressionBatchGD()
    obj.fit(X, y, plot=False)
    
    dw = obj.compute_gradient(X, y, weight)
    assert dw.shape == weight.shape, "Shapes don't match for compute_gradient() function"

@pytest.mark.parametrize("num_points, num_features",
    [(1,0), (100,0), (100,1), (100,2), (1000,4), (1000,6)]
)
def test_multiplication_11(num_points, num_features):
    np.random.seed(2024)
    
    X = np.random.rand(num_points, num_features+1 )
    X[:, 0] = 1 # bias term
    weights = np.random.rand(num_features+1,1)
    y = np.matmul(X, weights)
    
    obj = LinearRegressionBatchGD(max_epochs=1000, learning_rate=0.03)
    obj.fit(X, y, plot=False)
    
    y_hat = obj.predict(X)
    
    assert np.allclose(weights, obj.weights, atol=1e-02), f"Weights are not computed correctly. Difference:{np.sum(np.abs(y-y_hat))}"
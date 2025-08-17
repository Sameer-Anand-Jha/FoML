import pytest
from closedForm import LinearRegressionClosedForm
import numpy as np

def test_obj_init():
    obj = LinearRegressionClosedForm()

def test_fit_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    obj = LinearRegressionClosedForm()
    obj.fit(X, y)

def test_predict_function():
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    obj = LinearRegressionClosedForm()
    obj.fit(X, y)
    
    y_hat = obj.predict(X)
    assert y_hat.shape == y.shape, "Shapes don't match for predict() function"

@pytest.mark.parametrize("num_points, num_features",
    [(1,0), (100,0), (10,1), (10,2), (10,5), (100,1), (100,5), (1000,4)]
)
def test_multiplication_11(num_points, num_features):
    np.random.seed(2024)
    
    X = np.random.rand(num_points, num_features+1 )
    X[:, 0] = 1 # bias term
    weights = np.random.rand(num_features+1,1)
    y = np.matmul(X, weights)
    
    obj = LinearRegressionClosedForm()
    obj.fit(X, y)
    
    assert np.allclose(weights, obj.weights, atol=1e-03), "Weights are not computed correctly"
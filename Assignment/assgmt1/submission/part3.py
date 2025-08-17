import pandas as pd
import numpy as np
from IPython.display import display
from batchGradientDescent import LinearRegressionBatchGD
from closedForm import LinearRegressionClosedForm
from itertools import combinations

def laod_dataset(filepath):
    dataset = pd.read_csv(filepath)
    return dataset

def balance_data(dataset):
    class_counts = dataset['score'].value_counts()
    min_class_count = class_counts.min()

    balanced_dfs = []
    for class_label in class_counts.index:
        class_df = dataset[dataset['score'] == class_label]
        sampled_df = class_df.sample(n=min_class_count, random_state=10)
        balanced_dfs.append(sampled_df)

    dataset = pd.concat(balanced_dfs)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset

def extract_X_Y_ids(dataset):
    if 'score' in  dataset.columns:
        y = dataset['score'].to_numpy().reshape(-1, 1)
        X = dataset.iloc[:, 1:-1].to_numpy()
    else:
        y = None
        X = dataset.iloc[:, 1:].to_numpy()
    ids = dataset['ID'].to_numpy()
    
    return X, y, ids

def sigmoid(X):
    return np.exp(X)/(1+np.exp(X))

def preprocess_data(X):
    # mean normalization
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X  = (X - means)/(stds)

    # feature engineering
    n_cols = X.shape[1]
    combs_2 = list(combinations(range(n_cols), 2))
    interaction_features = []

    for comb in combs_2:
        interaction_term = X[:, comb[0]] * X[:, comb[1]]
        interaction_features.append(interaction_term)

    interaction_features = np.array(interaction_features).T
    X = np.hstack([X, X**2, X**3, np.cos(X), np.sin(X), sigmoid(X), 
                   sigmoid(X**2), sigmoid(np.cos(X)), sigmoid(np.sin(X)),
                   sigmoid(np.cos(X)**2),interaction_features])
    
    return X

# train-val splitting
def train_val_split(X, y, ids, train_ratio):
    train_size = int(len(X)*train_ratio)
    X_train = X[:train_size]
    y_train = y[:train_size]
    ID_train = ids[:train_size]

    X_val = X[train_size:]
    y_val = y[train_size:]
    ID_val = y[train_size:]

    return X_train, y_train, ID_train, X_val, y_val, ID_val

class RidgeRegressionClosedForm(LinearRegressionClosedForm):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def fit(self, X, y):
        # ridge
        self.weights = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ (X.T @ y)

# accuracy
accuracy = lambda y_hat, y_true: sum(1 for x, y in zip(y_hat, y_true) if x == y) / len(y_true)

train_ratio = 0.9

dataset = laod_dataset('train.csv')
X, y, ids = extract_X_Y_ids(dataset)

# closed form baseline performace
X_train, y_train, ID_train, X_val, y_val, ID_val = train_val_split(X, y, ids, train_ratio)
linear_reg_closed = LinearRegressionClosedForm()
linear_reg_closed.fit(X_train, y_train)
y_hat = linear_reg_closed.predict(X_val)
print(f"baseline performace (closed form - RMSE): { LinearRegressionBatchGD().compute_rmse_loss(X_val, y_val, linear_reg_closed.weights)}")
print(f"Validation Accuracy: {accuracy(y_hat, y_val)}")

# performance with feature engineering
# dataset = balance_data(dataset)   # commenting-out because it reduces performance
X, y, ids = extract_X_Y_ids(dataset)
X = preprocess_data(X)
X_train, y_train, ID_train, X_val, y_val, ID_val = train_val_split(X, y, ids, train_ratio)
print(f"Transformed feature space: {X.shape[1]}\nNumber of training samples: {X_train.shape[0]}")

# hyperparameter tuning
alphas = np.logspace(-3, 3, 100)
best_alpha = None
best_rmse = float('inf')

# grid search
for alpha in alphas:
    ridge_reg_closed = RidgeRegressionClosedForm(alpha=alpha)
    ridge_reg_closed.fit(X_train,y_train)
    y_hat = np.clip(np.round(ridge_reg_closed.predict(X_val)), 0, 4)
    rmse = LinearRegressionBatchGD().compute_rmse_loss(X_val, y_val, ridge_reg_closed.weights)
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

best_model = RidgeRegressionClosedForm(alpha=alpha)
best_model.fit(X_train, y_train)
y_hat = np.clip(np.round(ridge_reg_closed.predict(X_val)), 0, 4)
loss_val = LinearRegressionBatchGD().compute_rmse_loss(X_val, y_val, ridge_reg_closed.weights)
loss_train = LinearRegressionBatchGD().compute_rmse_loss(X_train, y_train, ridge_reg_closed.weights)
print(f"Improved Closed Form performance:\nTraining Loss: {loss_train} \t Validation loss: {loss_val}")
print(f"Validation Accuracy: {accuracy(y_hat, y_val)}")

# testing    
test_ds = laod_dataset('test.csv')
X_test, _, test_ids = extract_X_Y_ids(test_ds)
X_test = preprocess_data(X_test)
y_hat_test = np.clip(np.round(ridge_reg_closed.predict(X_test)), 0, 4)
y_hat_test = np.column_stack((test_ids, y_hat_test))
np.savetxt('kaggle.csv', y_hat_test, delimiter=',', header='ID,score', fmt=['%d', '%.1f'], comments='')

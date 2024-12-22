from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]

def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test

def calculate_covariance_matrix(X, Y=np.empty((0, 0))):
    if not Y.any():
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

def calculate_variance(X):
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - X.mean(axis=0)).T.dot(X - X.mean(axis=0)))
    return variance

def calculate_std_dev(X):
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev

def calculate_correlation_matrix(X, Y=np.empty([0])):
    covariance_matrix = calculate_covariance_matrix(X, Y)
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance_matrix, std_dev_X.dot(std_dev_y.T))
    return np.array(correlation_matrix, dtype=float)

class PCA():

    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.k = 2
    def transform(self, X):
        covariance = calculate_covariance_matrix(X)

        self.eigen_values, self.eigen_vectors = np.linalg.eig(covariance)

        idx = self.eigen_values.argsort()[::-1]
        eigenvalues = self.eigen_values[idx][:self.k]
        eigenvectors = self.eigen_vectors[:, idx][:, :self.k]

        X_transformed = X.dot(eigenvectors)

        return X_transformed

def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_trans = PCA().transform(X)

    x1 = X_trans[:, 0]
    x2 = X_trans[:, 1]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    plt.legend(class_distr, y, loc=1)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

if __name__ == "__main__":
    main()

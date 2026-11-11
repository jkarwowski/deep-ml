import numpy as np

def sigmoid(x: float) -> float:
    return 1/(1 + np.exp(-x))

def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    return np.mean((y - yhat)**2).round(4)

def dall(X: np.ndarray, weights: np.ndarray, bias: float, y: np.ndarray):
    z = (X @ weights) + bias
    yhat = sigmoid(z)
    n = yhat.shape[0]
    # print(((yhat - y) * (1 - yhat)).shape, X.shape)
    dweights = 2/n * ((yhat - y) * yhat * (1 - yhat)) @ X
    dbias = 2/n * (yhat - y) @ (yhat * (1 - yhat))
    return yhat, dweights, dbias


def train_neuron(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, lr: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    mse_values = []
    for e in range(epochs):
        yhat, update_w, update_b = dall(X, weights, bias, y)
        mse_values.append(mse(yhat, y))
        weights -= lr * update_w
        bias -= lr * update_b

    return weights.round(4).tolist(), bias.round(4).tolist(), mse_values
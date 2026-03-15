import numpy as np


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_sgd(XX, y, n_iterations, batch_size, learning_rate):
    n, d = XX.shape
    X = np.concatenate((np.ones((XX.shape[0], 1), dtype=float), XX), axis=1)
    classes = np.unique(y)
    K = len(classes)

    theta = np.zeros((d+1, K))

    # 1-0 labels
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1

    for _ in range(n_iterations):

        idx = np.random.choice(n, batch_size)
        Xb = X[idx]
        Yb = Y[idx]

        scores = Xb @ theta
        probs = softmax(scores)

        grad = Xb.T @ (probs - Yb) / batch_size

        theta -= learning_rate * grad

    return theta



def predict_softmax(X, theta):
    scores = X @ theta[1:] + theta[0]
    return np.argmax(scores, axis=1)

def predict_proba_softmax(X, theta):
    return softmax(X @ theta[1:] + theta[0])
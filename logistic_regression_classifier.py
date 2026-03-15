import numpy as np
import random as rnd


def randomizer (num_of_samples : int, batch_size : int) : # Use each epoch
    subsample = rnd.sample(range(num_of_samples), batch_size)
    return subsample



def sigmoid(z) :
    if z >= 20 :
        return 1.0
    elif z <= -20 :
        return 0.0
    else :
        if z >= 0:
            return 1/(1 + np.exp(-z))
        return np.exp(z)/(1 + np.exp(z))


def logit_grad(X,Y, theta) :
    # assuming theta.T@X(i) makes sense, and is a real number.
    mysum = np.zeros_like(theta)
    iterator = 0
    sample_size = Y.shape[0]
    # print(theta)
    while iterator < sample_size :

        mysum = mysum + (Y[iterator] - sigmoid(theta.T @ X[iterator]))*X[iterator]
        iterator += 1
    return mysum/sample_size


# %%
def logit_sgd(XX, y, n_iterations : int, batch_size : int, learning_rate : float) :
    # First prepare for an added constant. Add a column of ones to X.
    X = np.concatenate((np.ones((XX.shape[0], 1), dtype=float), XX), axis=1)

    iteration = 0
    n_samples = X.shape[0]
    current_theta = np.zeros(X.shape[1])
    # Number of iterations to be used for each use of gd.
    # print("starting sgd. epoch 1, current theta is 0")
    while iteration < n_iterations :
        indices = randomizer(n_samples , batch_size)

        X_train = X[indices]
        y_train = y[indices]

        # now do gradient descent :
        # print(f"epoch {epoch}, current theta is {str(current_theta)}")
        # print()
        gradient = logit_grad(X_train, y_train, current_theta)

        current_theta = current_theta + learning_rate * gradient

        epoch = epoch + 1
    # print()
    # print(f"loop finished. theta is {str(current_theta)}")
    return current_theta # This is t_0, t_1, t_2, ..., t_n


def predicted_probability(theta, X) :
    return sigmoid(theta[0] + theta[1:].T @ X)

def predicted_label(theta, X) :
    if predicted_probability(theta, X) > 0.5 :
        return 1
    return 0
import numpy as np
import random as rnd
import grad_desc as gd


# SGD
# INPUT : Data Matrix, Loss fn (assume convex for minimization!), batch size, epochs
# INITIALIZE : make `epochs`-many random partitions of full data, each batch of size `batch_size`-size.
# PROCESS : Each epoch : Each batch induces an error function.
# PROCESS : Perform one round of gd (on the restricted set of coordinates of theta)
# UPDATE : theta_i(t+1) = theta_i(t) - alpha*(normalized_gradient), only some theta_i get updated.
# STOP : Number of iterations reaches specified max

def randomizer (num_of_samples : int, batch_size : int) : # Use each epoch
    subsample = rnd.sample(range(num_of_samples), batch_size)
    return subsample

def ols_error_function(X,y) : # Returns a function that will take "Data", that is [X,y]
    def loss (theta) :
        return 0.5 * np.sum((X @ theta - y)**2)
    return loss



def sgd(X, y, n_epochs : int, batch_size : int, learning_rate : float) :
    epoch = 0
    n_samples = X.shape[0]
    current_theta = np.zeros(X.shape[1])
    n_iters = 200 # Number of iterations to be used for each use of gd.
    while epoch < n_epochs :
        indices = randomizer(n_samples , batch_size)

        X_train = X[indices]
        y_train = y[indices]

        error = ols_error_function(X_train,y_train)
        # now do gradient descent :
        gradient = gd.gd(error, current_theta, n_iters, learning_rate)
        g_norm = np.sum(gradient**2)
        if g_norm < 1e-6 :
            break
        else :
            gradient /= g_norm
            current_theta = current_theta - learning_rate * gradient
        epoch = epoch + 1
    return current_theta
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

# %%
def randomizer (num_of_samples : int, batch_size : int) : # Use each epoch
    subsample = rnd.sample(range(num_of_samples), batch_size)
    return subsample


# %%
def ols_error_function(X,y) : # Returns a function that will take "Data", that is [X,y]
    def loss (theta) :
        return np.sum((y - X @ theta)**2) # This X will have to have an extra column of 1s.
    return loss


# %%
def sgd(XX, y, n_epochs : int, batch_size : int, learning_rate : float) :
    # First prepare for an added constant. Add a column of ones to X.
    X = np.concatenate((np.ones((XX.shape[0], 1), dtype=float), XX), axis=1)

    epoch = 0
    n_samples = X.shape[0]
    current_theta = np.zeros(X.shape[1])
    # Number of iterations to be used for each use of gd.
    # print("starting sgd. epoch 1, current theta is 0")
    while epoch < n_epochs :
        indices = randomizer(n_samples , batch_size)

        X_train = X[indices]
        y_train = y[indices]

        error = ols_error_function(X_train,y_train)
        # now do gradient descent :
        # print(f"epoch {epoch}, current theta is {str(current_theta)}")
        # print()

        current_theta = gd.gd(error, current_theta, 1, learning_rate)

        epoch = epoch + 1
    # print()
    # print(f"loop finished. theta is {str(current_theta)}")
    return current_theta # This is t_0, t_1, t_2, ..., t_n
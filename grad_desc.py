import numpy as np


# There is scope for making the numerical derivative better. Not the time for it.
def numerical_derivative(f, x, axis, diff_step = 0.00001):
    if axis >= x.shape[0] :
        raise IndexError(f"Can't differentiate wrt x_{axis+1} when x has shape {x.shape}")
    else :
        basis_vector = np.zeros_like(x)
        basis_vector[axis] = diff_step

        left = (f(x) - f(x - basis_vector))/diff_step

        right = (f(x + basis_vector) - f(x))/diff_step

        return 0.5 * (left + right)

#-------------------------------------------------------------------------------------------------

# Takes a function and a point
# Returns an array (same shape as input vector) containing partial derivatives evaluated at the point
def grad(f, x) :
    gg = np.zeros(x.shape)
    # print(f"x has length {gg.shape[0]}")
    for i in range(x.shape[0]) :
        gg[i] = numerical_derivative(f, x, i)
        # print(f"using partial deriv {numerical_derivative(f, x, i)}")
    return gg

#-------------------------------------------------------------------------------------------------

# Takes a function, initial point, number of iterations and step size as input
# Performs gradient descent, returns an estimate of a local arg_min(f), given the starting point.
def gd(f, init_x, max_iters : int, step_size : float) :
    zero_init_vector = np.zeros_like(init_x)
    current_vector = init_x

    for i in range(max_iters):
        gradient = grad(f, current_vector)
        if np.linalg.norm(gradient) < 0.000001 :
            break
        gradient /= np.linalg.norm(gradient)
        current_vector = current_vector - step_size * gradient
        gradient = zero_init_vector
    return current_vector

#-------------------------------------------------------------------------------------------------
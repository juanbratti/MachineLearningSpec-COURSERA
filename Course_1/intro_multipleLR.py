import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(X : np.ndarray, y : np.ndarray, w : np.ndarray, b : float) -> float:
    """
    Computes the cost function for linear regression.

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar)    : model parameter

    Returns:
        cost (scalar): cost
    """

    # amount of values
    m = X.shape[0]
    cost = 0.0
    predictions = []
    # calculate predictions
    for i in range(m):
        predictions = np.append(predictions, np.dot(X[i],w) + b)

    cost = np.sum((predictions - y)**2/(2*m))
  
    return cost

def compute_derivatives(X:np.ndarray, y:np.ndarray, w:np.ndarray ,b:float) -> float:

    """
    Computes the derivatives part of the gradient descent algorithm

    Args:
        x (ndarray (m,n)): Data, m examples
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar)    : model parameters

    Returns:
        dj_dw : (ndarray (n))
        dj_b : scalar
    """

    m,n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db += err

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw

def gradient_descent(X : np.ndarray, y : np.ndarray, w_in : np.ndarray, b_in : float, cost_function, gradient_function, alpha : float, num_iters : float) -> float:
    """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters  
        b_in (scalar)       : initial model parameter
        cost_function       : function to compute cost
        gradient_function   : function to compute the gradient
        alpha (float)       : Learning rate
        num_iters (int)     : number of iterations to run gradient descent
        
        Returns:
        w (ndarray (n,)) : Updated values of parameters 
        b (scalar)       : Updated value of parameter 
    """

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        J_history.append(cost_function(X, y, w, b))

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
        
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_derivatives, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0,500.0])

def compute_cost(x : np.ndarray, y : np.ndarray, w : float, b : float) -> float:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters

    Returns:
        total_cost (float): The cost of using w,b as the parameters 
        for linear regression to fit the data points in x and y
    """

    # amount of values
    m = len(x)

    # calculating predictions
    predictions = w*x + b #f_wb
    total_cost = (np.sum((predictions - y)**2))/(2*m)

    return total_cost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()


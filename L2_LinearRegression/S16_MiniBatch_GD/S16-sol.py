def MSEStep(X, y, W, b, learn_rate = 0.001):
    """
    This function implements the gradient descent step for squared error as a
    performance metric.
    
    Parameters
    X : array of predictor features
    y : array of outcome values
    W : predictor feature coefficients
    b : regression function intercept
    learn_rate : learning rate

    Returns
    W_new : predictor feature coefficients following gradient descent step
    b_new : intercept following gradient descent step
    """
    
    # compute errors
    y_pred = np.matmul(X, W) + b
    error = y - y_pred
    
    # compute steps
    W_new = W + learn_rate * np.matmul(error, X)
    b_new = b + learn_rate * error.sum()
    return W_new, b_new



#============================================================
import numpy as np
np.random.seed(42)
data = np.loadtxt('data.csv', delimiter = ',')
X = data[:,:-1]
y = data[:,-1]

#regression_coef = miniBatchGD(X, y)
batch_size = 20
learn_rate = 0.005 
num_iter = 25
n_points = X.shape[0]
W = np.zeros(X.shape[1]) # coefficients
b = 0 # intercept


regression_coef = [np.hstack((W,b))]
for _ in range(num_iter):
    batch = np.random.choice(range(n_points), batch_size)
    X_batch = X[batch,:]
    y_batch = y[batch]
    #W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
    y_pred = np.matmul(X_batch, W) + b
    error = y_batch - y_pred
    W = W + learn_rate * np.matmul(error, X_batch)
    b = b + learn_rate * error.sum()
    regression_coef.append(np.hstack((W, b)))


#============================================================


'''answer
regression_coef = [np.hstack((W,b))]
for _ in range(num_iter):
    batch = np.random.choice(range(n_points), batch_size)
    X_batch = X[batch,:]
    y_batch = y[batch]
    #W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
    y_pred = np.matmul(X, W) + b
    error = y - y_pred
    W = W + learn_rate * np.matmul(error, X)
    b = b + learn_rate * error.sum()
    regression_coef.append(np.hstack((W, b)))


regression_coef:
[array([0., 0.]), array([0.12341158, 0.2049234 ]), array([0.07713629, 0.37119546]), array([0.13991386, 0.52422169]), array([0.14017334, 0.66281191]), array([0.15514756, 0.77699134]), array([0.15039492, 0.88136108]), array([0.19857266, 0.99942708]), array([0.26441435, 1.1012103 ]), array([0.284643 , 1.1749923]), array([0.31173667, 1.2578259 ]), array([0.33257141, 1.29953887]), array([0.37719351, 1.35343161]), array([0.38658478, 1.41235015]), array([0.37639877, 1.48358489]), array([0.40449639, 1.51995864]), array([0.38520041, 1.57841031]), array([0.41715857, 1.63043345]), array([0.41430519, 1.66112077]), array([0.39842119, 1.68996465]), array([0.39392107, 1.71004375]), array([0.42072954, 1.72423628]), array([0.42726705, 1.74816547]), array([0.4565849 , 1.77112757]), array([0.45097337, 1.78769497]), array([0.4477919 , 1.82878844])]
'''

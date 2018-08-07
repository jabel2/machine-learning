
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

get_ipython().run_line_magic('matplotlib', 'inline')

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    #s = np.divide(1, 1+np.exp(-z))
    s = 1 / (1 + np.exp(-z))
    
    
    return s

#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    
    w = np.zeros((dim, 1))
    b = 0
   

    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

#dim = 2

#w, b = initialize_with_zeros(dim)
#print ("w = " + str(w))
#print ("b = " + str(b))

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    
    m = X.shape[1]  # compute the number of training data
    
    
    # FORWARD PROPAGATION (FROM X TO COST)
   
    #compute z = wTranspose * X+b
    Z = np.matmul(w.T, X) + b
    
    #A = sigmouid (Z) where z = wTranspose * X+b
    #A = Yhat
    
    # compute activation (1 x M vector)
    A = sigmoid(Z)     
    
    # compute cost
    # 1/m SUM L(Yhat, y)
    # L(a,y) = - (y log a + (1-y) log (1-a))
    
    L = -((np.matmul(Y, np.log(A.T))) + np.matmul((1-Y), np.log(1-A.T)))
    
    # compute cost by doing 1/m * L
    cost = np.divide(L,m)
   
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
   
    # for one data point dw = (a-y)x1
    dw1 = np.matmul(X,(A-Y).T)
    dw = np.divide(dw1, m)
    
    # for one data point db = a-y
    db1 = np.sum(np.subtract(A, Y))
    db = np.divide(db1,m)

   

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w,b,X,Y)
        
        # Retrieve derivatives from grads dictionary
        dw = grads['dw'] 
        db = grads['db']        
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    
     #compute z = wTranspose * X+b
    Z = np.matmul(w.T, X) + b
    #A = sigmouid (Z) where z = wTranspose * X+b
    #A = Yhat
    # compute activation (1 x M vector)
    A = sigmoid(Z)
   
    
    #for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
    Y_prediction = np.around(A)
    #Y_prediction = np.where(A > 0.5, 1, 0)
        
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#print ("predictions = " + str(predict(w, b, X)))

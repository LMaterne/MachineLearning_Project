#additional functions
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return (1.0/n) *(res.T.dot(res))

def R2(data, model):
    """
    calculate the R2 score function
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    numerator = (res.T.dot(res))
    res1 = (data - np.mean(data))
    denominator = (res1.T.dot(res1))
    #numerator = MSE(data, model)
    #denominator = MSE(data, np.mean(data))
    return 1.0 - numerator/denominator

def plot_it(x,y,model,franke_data):
    '''
    This is a function to plot the x y and z data
    Inputs: z-axis data
    
    This function takes is the zdata only as x and y are assumed to stay the same
    the zdata this could be from the franke function or the model,
    you just need to specify
    
    Outputs: This function returns nothing    
    
    '''    
    fig = plt.figure()    
    ax = plt.axes(projection='3d')
    
    
    ax.plot_trisurf(x, y, model, cmap='viridis', edgecolor='none')
    ax.scatter(x,y,franke_data)
    
   
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The franke function as scatter and the model fit')
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def kfold(x,y):
    '''This is a function to resample the data into training data for the model and test data.
    Inputs: x data from the full set
            y data from the full set
    Outputs: its outputs the xtrain, xtest, ytrain, ytest in this order
            xtrain will have 0.66 of the full data set
            xtest will have 0.33 of the full data set
    
    '''
    n = np.shape(x)[0]
    meanX = np.zeros(n)
    meanY = np.zeros(n)
    k = 0
    while k < n:
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.66)
        meanX[k] = np.mean(xtrain)
        meanY[k] = np.mean(ytrain)
        
        
        k=k+1
    return meanX, meanY

    
#additional functions
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split

from imageio import imread

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

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
    Inputs: x: the generated x points
            y: the generated y points
            model: the model we are testing
            franke_data : the data from the frankefunction
    '''
    ax = plt.axes(projection='3d')

    # plots scatter and trisurf
    ax.plot_trisurf(x, y, model, cmap='viridis', edgecolor='none')
    ax.scatter(x,y,franke_data)

    #set the axis labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The sample data plotted as a scatter & the model plotted as a trisurf')

    plt.show()


def load_terrain(imname):
# Load the terrain
    terrain = imread('{:}.tif'.format(imname))
    terraindata = scipy.misc.imread('{:}.tif'.format(imname))
# Show the terrain
    plt.figure()
    plt.title('Terrain over Norway')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
# return the terrain for plotting and the data using scipy
    return terrain,terraindata

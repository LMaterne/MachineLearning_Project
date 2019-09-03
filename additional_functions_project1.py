#additional functions
import numpy as np
import subprocess
import matplotlib.pyplot as plt 

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def generate_sample(n, mean = 0, var = 1):
    """
    Generates (n,3) samples [x,y,z], where x,y are uniform random numbers [0,1) 
    and z = f(x,y) + eps with f the Franke function and eps normal distributed with mean and var
    """
    x, y = np.random.rand(2,n)
    z = FrankeFunction(x,y) + np.sqrt(var)*np.random.randn(n) + mean
    return np.array([x, y, z])

def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return 1.0/n *res.T.dot(res)

def R2(data, model):
    """
    calculate the R2 score function
    """
    numerator = MSE(data, model)
    denominator = MSE(data, np.mean(data))
    return 1 - numerator/denominator

def variance_bias_tradeoff(comp, MSE, Var, Bias, filename, filepath = 'results', font = 18, title=None):
    try:
        subprocess.call(['cd',filepath])
    except:
        subprocess.call(['mkdir', filepath])

    plt.figure(figsize=(10,10))
    plt.title(title, fontsize = font)
    plt.plot(comp, MSE, label ='MSE')
    plt.plot(comp, Var, lable ='Model Variance')
    plt.plot(comp, Bias, label = 'Model Bias')
    plt.xlabel("Model Complexity; Order of Polynomial", fonstize = font)
    plt.ylabel("Deviations", fonstize = font)
    plt.legend(loc='best', fonstize = font)
    plt.savefig(filepath + filename)
    
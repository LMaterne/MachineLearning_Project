import numpy as np 

def matDesign (dataSet,order,indVariables):
    '''This is a function to set up the design matrix
    the inputs are :dataSet, the n datapoints, x and y data in a nx2 matrix
                    order, is the order of the coefficients, 
                    indVariables, the number of independant variables or predictors
                    
    i.e if order = 3 and indVariables = 1, then the number of coefficients THIS function will create is 4. (1 x x**2 x**3)
    or  if order = 2 and indVariables = 2, then the number of coefficients THIS function will create is 6. (1 x y xy x**2 y**2) 
    
    IMPORTANT NOTE: this works only for indVariables = 2 at the moment
    
    the outputs are X
    '''

    # if statement for the case with one independant variable
    if indVariables == 1:
        coefficients = int(order + 1)
        
        # set up the Design matrix
        n = np.int(np.size(dataSet))
        matX = np.zeros((n,coefficients))
    
        # loop through all the other columns as powes of dataSet
        i = 0 #counter
        while i < coefficients:
            matX[:,i] = (dataSet[i])**i
            i=i+1
        
        
    ###########################################################################################################
    
    # if statement for the case with two independant variables
    
    if (indVariables == 2):
        coefficients = int((order + 1)*(order + 2)/2)
        
    # set up the Design matrix
        #find the number of rows in dataSet
        rows,columns = np.hsplit(dataSet,2) # this is a messy way to find n 
        n = np.int(np.size(rows))
        
        matX = np.zeros((n,coefficients))
        #print(matX)
        
        # loop through all the other columns as powes of xDataset
        # THIS IS NOT FINISHED AS THERE IS NO LOOP
        matX[:,0] = 1
        matX[:,1] = dataSet[:,0]
        matX[:,2] = dataSet[:,1]
        matX[:,3] = (dataSet[:,0])*(dataSet[:,1])
        matX[:,4] = (dataSet[:,0])**2
        matX[:,5] = (dataSet[:,1])**2
        

    return matX

def linReg(data, design):
    """
    returns the estimated parameters of an OLS
    """
    inverse = np.linalg.inv(design.T.dot(design))
    par = inverse.dot(design.T).dot(data)
    return par

def ridgeReg(data, design, lam = 0.1):
    """
    returns the estimated parameters of an Ridge Regression with 
    regularization parameter lambda
    """
    #creating identity matrix weighted with lam
    diag = lam * np.ones(design.shape[1])
    inverse = np.linalg.inv(design.T.dot(design) + np.diag(diag))
    par = inverse.dot(design.T).dot(data)
    return par

#additional functions
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
    return 1.0/n *res.T.dot(res)

def R2(data, model):
    """
    calculate the R2 score function
    """
    numerator = MSE(data, model)
    denominator = MSE(data, np.mean(data))
    return 1 - numerator/denominator

def generate_sample(n, mean = 0, var = 1):
    """
    Generates (n,3) samples [x,y,z], where x,y are uniform random numbers [0,1) 
    and z = f(x,y) + eps with f the Franke function and eps normal distributed with mean and var
    """
    x, y = np.random.rand(2,n)
    z = FrankeFunction(x,y) + np.sqrt(var)*np.random.randn(n) + mean
    return np.array([x, y, z])

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

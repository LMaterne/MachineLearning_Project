import Poly2DFit
import numpy as np
import pandas as pd

def main():
    
    
    n = 100
    order = 6
    
    #Initialize a dataframe to store the results:
    col1 = []
    for j in range(0,order):        
        for k in range(0,j+1):    
                name = 'x%s y%s'% (j-k,k)
                col1.append(name)        
            
    # initialuse the name of rows and columns        
    ind = ['model_pow_%d'%i for i in range(1,order)]
    col = ['mse', 'r2','bias', 'variance']    
    table_of_info = pd.DataFrame(index=ind, columns=col)
    coef_matrix = pd.DataFrame(index=ind, columns=col1)
    
    #loop for creating fits for many orders
    for i in range(0,order):
        #create fit object
        fit_object = Poly2DFit.Poly2DFit()
    
        #generate data with noise: mean 0, var =1
        #fit_object.generateSample(n)
        fit_object.generateKfold(n)
        
        #returns the fitted parameters and their variance
        par, par_var = fit_object.run_fit( i, 'OLS'  )
    
        #evaluate model, return x,y points and model prediction
        x, y, fit = fit_object.evaluate_model()
        
        table_of_info.iloc[i-1,0] = fit_object.mse # this is an error don't know why i need this -1
    
        table_of_info.iloc[i-1,1] = fit_object.r2
       
        table_of_info.iloc[i-1,2] = fit_object.bias
        table_of_info.iloc[i-1,3] = fit_object.variance
        
        num_of_p = int((i+1)*(i+2)/2)
        
        
        for m in range (0,num_of_p):
            coef_matrix.iloc[i-1,m] = fit_object.par[m]
        
        
    #stores information about the fit in ./Test/OLS/first_OLS.txt
    #fit_object.store_information('Test/OLS','first_OLS')
    pd.options.display.float_format = '{:,.2g}'.format
    print (table_of_info)
    print (coef_matrix)
    

    #stores information about the fit in ./Test/OLS/first_OLS.txt
    #fit_object.store_information('Test/OLS','first_OLS')

    #same procedure but now as ridge wit lambda = 0.1
    #par, par_var = fit_object.run_fit( order, 'RIDGE', 0.1  )
    
    #evaluate model, return x,y points and model prediction
    #x, y, fit = fit_object.evaluate_model()
    
    #stores information about the fit in ./Test/OLS/first_OLS.txt
    #fit_object.store_information('Test/RIDGE','first_RIDGE')
    
    
    #plot the data and model
    fit_object.plot_function()
    
     
    
    
if __name__ == "__main__":
    main()
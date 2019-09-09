import Poly2DFit
import numpy as np
import pandas as pd

def main():
    
    
    n = 100
    #order = 7
    #Initialize a dataframe to store the results:
    
#    for j in range(0,8):
#        for k in range(0,j):
#            col = ['mse','r2'] + ['coef_x_%d'% (j-k) ,'coef_y_%d'%k ]
    ind = ['model_pow_%d'%i for i in range(1,10)]
    col = ['mse','r2','bias', 'variance']
    coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
    
    for i in range(1,10):
    #create fit object
        fit_object = Poly2DFit.Poly2DFit()
    
    #generate data with noise: mean 0, var =1
        fit_object.generateSample(n)
    
    #returns the fitted parameters and their variance
        par, par_var = fit_object.run_fit( i, 'OLS'  )
    
    #evaluate model, return x,y points and model prediction
        x, y, fit = fit_object.evaluate_model()
        
        coef_matrix_simple.iloc[i-1,0] = fit_object.mse
        coef_matrix_simple.iloc[i-1,1] = fit_object.r2
        coef_matrix_simple.iloc[i-1,2] = fit_object.bias
        coef_matrix_simple.iloc[i-1,3] = fit_object.variance
        
    #stores information about the fit in ./Test/OLS/first_OLS.txt
    #fit_object.store_information('Test/OLS','first_OLS')
    pd.options.display.float_format = '{:,.2g}'.format
    print (coef_matrix_simple)
    
    

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
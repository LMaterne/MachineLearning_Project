import Poly2DFit
import numpy as np


def main():
    n = 100
    order = 7
    
    #create fit object
    fit_object = Poly2DFit.Poly2DFit()
    
    #generate data with noise: mean 0, var =1
    fit_object.generateSample(n)
    
    #returns the fitted parameters and their variance
    par, par_var = fit_object.run_fit( order, 'OLS'  )
    
    #evaluate model, return x,y points and model prediction
    x, y, fit = fit_object.evaluate_model()

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
    
    # new instance of the class
    fit_object1 = Poly2DFit.Poly2DFit()
    #now do the same as above but using k fold cross validation
    fit_object1.generateKfold(n)
    par, par_var = fit_object1.run_fit( order, 'OLS'  )
    fit_object1.evaluate_model()
    fit_object1.plot_function()
     
    
    
if __name__ == "__main__":
    main()
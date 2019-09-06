import Poly2DFit
import numpy as np


def main():
    n = 10
    order =3

    #creat fit object
    fit_object = Poly2DFit.Poly2DFit()
    #generate data with noise: mean 0, var =1
    fit_object.generateSample(n)
    
    #fit_object.generate_split_Sample(n)
    #print('this is the split x and y', fit_object.x,fit_object.y)
     
    
    #run polynomial fit of order 3, OLS type
    #returns the fitted parameters and theire variance
    par, par_var = fit_object.run_fit( order, 'OLS'  )
    #evaluate model, return x,y points and model prediction
    x, y, fit = fit_object.evaluate_model()
    
    print(fit_object.x)  
    #stores information about the fit in ./Test/OLS/first_OLS.txt
#    fit_object.store_information('Test/OLS','first_OLS')

    #same procedure but now as ridge wit lambda = 0.1
    par, par_var = fit_object.run_fit( order, 'RIDGE', 0.1  )
    #evaluate model, return x,y points and model prediction
    x, y, fit = fit_object.evaluate_model()
    #stores information about the fit in ./Test/OLS/first_OLS.txt
#    fit_object.store_information('Test/RIDGE','first_RIDGE')
    
    print('zdata from franke is shape', np.shape(fit_object.data))
    fit_object.plot_function(fit_object.data)
    
    
    
    
if __name__ == "__main__":
    main()
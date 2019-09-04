import Poly2DFit
import numpy as np

def main():
    n = 10000
    order =3

    #creat fit object
    fit_object = Poly2DFit.Poly2DFit()
    #generate data with noise: mean 0, var =1
    fit_object.generateSample(n)
    #run polynomial fit of order 3, OLS type
    #returns the fitted parameters and theire variance
    par, par_var = fit_object.run_fit( order, 'OLS'  )
    #evaluate model, return x,y points and model prediction
    x, y, fit = fit_object.evaluate_model()
    #stores information about the fit in ./Test/OLS/first_OLS.txt
    fit_object.store_information('Test/OLS','first_OLS')

    #same procedure but now as ridge wit lambda = 0.1
    par, par_var = fit_object.run_fit( order, 'RIDGE', 0.1  )
    #evaluate model, return x,y points and model prediction
    x, y, fit = fit_object.evaluate_model()
    #stores information about the fit in ./Test/OLS/first_OLS.txt
    fit_object.store_information('Test/RIDGE','first_RIDGE')
    
if __name__ == "__main__":
    main()
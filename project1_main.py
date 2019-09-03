from additional_functions_project1 import generate_sample, variance_bias_tradeoff
from functions_project1 import run_fit, matDesign

import numpy as np

def main():
    n = 1000
    maxorder = 5

    x,y,z = generate_sample(n)

    p = np.arange(0,maxorder+1)
    fit = ['OLS', 'RIDGE']
    lam = [0.01, 0.1, 0.5, 1]

    mod_var_ridge, mod_bias_ridge, mse_ridge = np.zeros((3, maxorder+1, len(lam)))
    mse_ols, mod_var_ols, mod_bias_ols = np.zeros((3, maxorder +1))
    
    ####part a)
    #calc models
    for i,p in enumerate(p):
        design = matDesign([x,y], p, 2)
        for f in fit:
            if f != 'OLS':
                for j,l in enumerate(lam):
                    ret = run_fit(z, design, f, l, filepath='part_a_pol%i'%i)
                    mod_var_ridge[i,j] = ret['Model_Variance']
                    mod_bias_ridge[i,j] = ret['Model_Bias']
                    mse_ridge[i,j] = ret['MSE']
            else:
                ret = run_fit(z, design, f, filepath='part_a_pol%i'%i)
                mod_var_ols[i] = ret['Model_Variance']
                mod_bias_ols[i] = ret['Model_Bias']
                mse_ols[i] = ret['MSE']
    #plot models
    for f in fit:
        if f != 'OLS':
            for i, l in enumerate(lam):
                variance_bias_tradeoff(p, mse_ridge[:,i], mod_var_ridge[:,i],
                                         mod_bias_ridge[:,i], f + str(l),
                                         filepath='results_part_a',title='Lambda=%.2f'%l)

        else:
            variance_bias_tradeoff(p, mse_ols, mod_var_ols, mod_bias_ols, f + str(l),
                                    filepath='results_part_a')



if __name__ == "__main__":
    main()
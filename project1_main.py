import Poly2DFit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_stats(info, title = 'Regression Infos'):
    data = pd.DataFrame(columns = ['Complexity','Value', 'Metric'] )
    n = len(info['power'].to_numpy())
    for t in ['MSE', 'Bias', 'Variance']:
        dat = np.array( [info['power'].to_numpy(),  info[t.lower()].to_numpy(), [t for i in range(n)]])
        
        app = pd.DataFrame(dat.T, columns =['Complexity','Value', 'Metric'] )
        data = data.append(app)
    
    plt.title(title)
    sns.lineplot(x = 'Complexity', y= 'Value', hue = 'Metric', data = data, estimator = None)
    plt.show()

def benchmarking( regressiontype, n = 500, order = 7, lam = 0.1,
                 display_info = True, plot_info = True, plot_fit =False, save_file = False):
    
    #Initialize a dataframe to store the results:
    col1 = []
    for j in np.arange(0,order):        
        for k in np.arange(0,j+1):    
                name = 'x%s y%s'% (j-k,k)
                col1.append(name)        
            
    # initialuse the name of rows and columns        
    ind = ['model_pow_%d'%i for i in np.arange(0,order)]
    col = ['power','mse', 'r2','bias', 'variance']    
    table_of_info = pd.DataFrame(index=ind, columns=col)
    coef_matrix = pd.DataFrame(index=ind, columns=col1)
    
    #loop for creating fits for many orders
    for i in np.arange(0,order):
        #create fit object
        fit_object = Poly2DFit.Poly2DFit()
    
        #generate data with noise: mean 0, var =1
        fit_object.generateSample(n, 'split' , 5)
        
        #returns the fitted parameters and their variance
        par, par_var = fit_object.run_fit( i, regressiontype, lam )
    
        #evaluate model, return x,y points and model prediction
        x, y, fit = fit_object.evaluate_model()
        
        #save informaton
        table_of_info.iloc[i,0] = i
        table_of_info.iloc[i,1] = fit_object.mse 
        table_of_info.iloc[i,2] = fit_object.r2
        table_of_info.iloc[i,3] = fit_object.bias
        table_of_info.iloc[i,4] = fit_object.variance
        
        if plot_fit:
            #plot the data and model
            fit_object.plot_function()

#        if save_file:
             #stores information about the fit in ./Test/OLS/first_OLS.txt
#             fit_object.store_information(regressiontype, 'order_%i' % i)
        
        #find the number of parameters and then put these parameters in a table
        num_of_p = int((i+1)*(i+2)/2)
        for m in np.arange(0,num_of_p):
            coef_matrix.iloc[i,m] = fit_object.par[m]
        
   
    
    if display_info:
        pd.options.display.float_format = '{:,.2g}'.format
        print (table_of_info)
        print (coef_matrix)

    if plot_info:
        plot_stats(table_of_info, regressiontype +' Regression Info')
    
    return table_of_info


def main():
    
    toi_ols = benchmarking('OLS', 100, 5, save_file=True)
#    toi_ridge = benchmarking('RIDGE', 100, 5, save_file=True)
      
    
    
if __name__ == "__main__":
    main()
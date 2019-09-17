import Poly2DFit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def toi_append(data, info, regressiontype, lam, kFold):
    n = len(info['power'].to_numpy())
    app = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    for t in ['MSE', 'Bias', 'Variance']:
        dat = np.array([[regressiontype for i in range(n)],
                       [lam for i in range(n)],
                       [kFold for i in range(n)],
                       info['power'].to_numpy(),
                       info[t.lower()].to_numpy(),
                       [t for i in range(n)]])
        temp = pd.DataFrame(dat.T, columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
        
        app = app.append(temp)
            
    return  data.append(app)

def plot_stats(info, title = 'Regression Infos'):
    data = pd.DataFrame(columns = ['Complexity','Value', 'Metric'] )
    n = len(info['power'].to_numpy())
    for t in ['MSE', 'Bias', 'Variance']:
        dat = np.array( [info['power'].to_numpy(),  info[t.lower()].to_numpy(), [t for i in range(n)]])
        
        app = pd.DataFrame(dat.T, columns =['Complexity','Value', 'Metric'] )
        data = data.append(app)
    
    plt.title(title)
    sns.lineplot(x = 'Complexity', y= 'Value', hue = 'Metric', data = data, estimator = None)
    plt.ylim(0, 1.5 ) #for plot comparison
    plt.show()

def benchmarking( regressiontype, n = 500, order = 7, lam = 0.1, kfold = 0,
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
        fit_object.generateSample(n)

        if kfold != 0:
            fit_object.kfold_cross(i, regressiontype, lam, kfold )
        else:        
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

        if save_file:
             #stores information about the fit in ./Test/OLS/first_OLS.txt
             fit_object.store_information(regressiontype, 'order_%i' % i)
        
        #find the number of parameters and then put these parameters in a table
        num_of_p = int((i+1)*(i+2)/2)
        for m in np.arange(0,num_of_p):
            coef_matrix.iloc[i,m] = fit_object.par[m]
        
   
    
    if display_info:
        pd.options.display.float_format = '{:,.2g}'.format
        print (table_of_info)
        print (coef_matrix)

    if plot_info:
        title = regressiontype +' Regression Info'
        if regressiontype != 'OLS':
            title += ' $\lambda$ = %.2f' % lam
        plot_stats(table_of_info,title )
    
    return table_of_info


def main():

    ks = [0, 5, 10]
    lam = [0, 0.01, 0.1, 1]
    max_order = 6
    samples = 100

    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    for k in ks:
        temp = benchmarking('OLS', samples, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        for l in lam:
            temp = benchmarking('RIDGE', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)

    #filter for lam
    lam_filter =  (toi['lambda'] == 0) #(toi['lambda'] == 0.1 ) |
    #filter for Ridge
    ridge_filter = toi['Regression type'] == 'RIDGE'
    
    #compare kfold for different regressions
    g = sns. FacetGrid(toi[lam_filter], row ='Regression type', col='kFold', hue ='Metric')
    g.map(plt.plot, 'Complexity', 'Value')
    g.add_legend()
    g.savefig('ols_vs_ridge.pdf')

    #compare kfold and lambda fore ridge
    g = sns. FacetGrid(toi[ridge_filter], row ='lambda', col='kFold', hue ='Metric')
    g.map(plt.plot, 'Complexity', 'Value')
    g.add_legend()
    g.savefig('lam_vs_kfold.pdf')

      
    
    
if __name__ == "__main__":
    main()
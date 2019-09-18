import Poly2DFit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

def toi_append(data, info, regressiontype, lam, kFold):
    n = len(info['power'].to_numpy())
    app = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    for t in ['MSE', 'Bias', 'Variance', 'R2','MSE_train']:
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
    col = ['power','mse', 'r2','bias', 'variance', 'mse_train']
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
            fit_object.evaluate_model()

        #save informaton
        table_of_info.iloc[i,0] = i
        table_of_info.iloc[i,1] = fit_object.mse
        table_of_info.iloc[i,2] = fit_object.r2
        table_of_info.iloc[i,3] = fit_object.bias
        table_of_info.iloc[i,4] = fit_object.variance
        if kfold != 0:
            table_of_info.iloc[i,5] = fit_object.mse_train

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

    ks = [0, 1, 5, 10]
    lam = [10**(-5), 10**(-3), 10**(-1)]

    max_order = 10
    samples = 1800*3600

    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )

    ops = len(ks) * ( 2 * len(lam) + 1)
    one_part = 100 / ops #in%
    current_progress = 0

    for k in ks:
        current_progress += one_part
        print("Now: OLS; Progress", int(current_progress), "%")
        temp = benchmarking('OLS', samples, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        
        for l in lam:
            current_progress += one_part
            print("Now: RIDGE; Progress", int(current_progress), "%")
            temp = benchmarking('RIDGE', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)


            current_progress += one_part
            print("Now: LASSO; Progress", int(current_progress), "%")
            
            temp = benchmarking('LASSO', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'LASSO', l, k)
            
            
            
    #filter for lam
    lam_filter =  ((toi['lambda'] == 0) | (toi['lambda'] == 0.001)) & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #filter for Ridge
    ridge_filter = (toi['Regression type'] == 'RIDGE') & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #filter for lasso
    lasso_filter = (toi['Regression type'] == 'LASSO') & ((toi['Metric'] != 'R2') & (toi['Metric'] != 'MSE_train'))
    #R2 filter
    r2_filter = toi['Metric'] =='R2'
    #book filter
    book_filter = ((toi['Metric']=='MSE') | (toi['Metric']=='MSE_train')) &  ((toi['lambda'] == 0) | (toi['lambda'] == 0.001)) & (toi['kFold'] != 0)

    #compare kfold for different regressions
    g = sns. FacetGrid(toi[lam_filter], row ='Regression type', col='kFold', hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.add_legend()
    g.set_axis_labels('Polynom Order', 'MSE')
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig('comp_fit.pdf')

    #compare kfold and lambda for ridge
    g = sns. FacetGrid(toi[ridge_filter], row ='lambda', col='kFold', hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'MSE')
    g.add_legend()
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig('ridge_lam_vs_kfold.pdf')
    
    #compare kfold and lambda for lasso
    g = sns. FacetGrid(toi[lasso_filter], row ='lambda', col='kFold', hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'MSE')
    g.add_legend()
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig('lasso_lam_vs_kfold.pdf')
    
    #compare kfold and lambda fore ridge
    g = sns. FacetGrid(toi[r2_filter], row ='lambda', col='kFold', hue ='Regression type', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'R2')
    g.add_legend()
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig('r2.pdf')

    #make plot from book
    g = sns. FacetGrid(toi[book_filter], row ='Regression type', col='kFold', hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'MSE')
    g.add_legend()
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig('train_vs_test.pdf')


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")

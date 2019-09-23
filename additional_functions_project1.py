#additional functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import Poly2DFit

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return (1.0/n) *(res.T.dot(res))

def R2(data, model):
    """
    calculate the R2 score function
    """
    res = np.array(data - model)
    numerator = (res.T.dot(res))
    res1 = (data - np.mean(data))
    denominator = (res1.T.dot(res1))
    return 1 - numerator/denominator


def plot_it(x,y,model,franke_data):
    '''
    This is a function to plot the x y and z data
    Inputs: x: the generated x points
            y: the generated y points
            model: the model we are testing
            franke_data : the data from the frankefunction
    '''
    ax = plt.axes(projection='3d')

    # plots scatter and trisurf
    ax.plot_trisurf(x, y, model, cmap='viridis', edgecolor='none')
    ax.scatter(x,y,franke_data)

    #set the axis labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The sample data plotted as a scatter & the model plotted as a trisurf')

    plt.show()


def load_terrain(imname):
# Load the terrain
    terrain = imread('{:}.tif'.format(imname))
# Show the terrain
    plt.figure()
    plt.title('Terrain over Norway')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
# return the terrain data,
# which is a matrix with values corresponding to f(x,y) - height
#reducing terrain data
    N = len(terrain[0][::4]) # number of columns in reduced matrix
    n = len(terrain[::4][0]) # number of rows in reduced matrix
#print(n,N), print(np.shape(terrain))
    reduced  = np.zeros((n,N))

    for i in range(n-1):
        reduced[i] = terrain[i][::4]
# print(reduced)
    return reduced

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

def duplicate_axis(ax):
    l = ax.lines[0]
    iter_line = True
    it = 0
    x = l.get_xdata()
    y = l.get_ydata()
    it += 1
    while iter_line:
        try:
            l = ax.lines[it]
            xt = l.get_xdata()
            yt = l.get_ydata()
            it += 1
            x = np.vstack([x,xt])
            y = np.vstack([y,yt])
        except:
            iter_line = False
    return x, y

def plotting_mse(toi, row, col, filename, split = False, ylabel ='MSE', shary = True):
    """
    gives the table of informations, toi, to facetgrid (apply filter in call)
    spans dimension by col, row with toi column names
    saves as filename
    if split is True: store each subplot seperatly
    if shary is True: shared y axes (only in split)
    """
    max_order = toi['Complexity'].max()
    g = sns. FacetGrid(toi, row =row, col=col, hue ='Metric', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.add_legend()
    g.set_axis_labels('Polynom Order', ylabel)
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig(filename + '.pdf')

    if split:
        labels = toi['Metric'].unique()
        axes = g.axes.flat
        for i,ax in enumerate(axes):
            title =ax.get_title()
            xax = ax.get_xlim()
            yax =ax.get_ylim()
            x,y  = duplicate_axis(ax)

            f = plt.figure(figsize=(10,10))
            for k in range(len(x)):
                plt.plot(x[k], y[k], label = labels[k])
            plt.xlim(xax)
            if shary:
                plt.ylim(yax)
            plt.ylabel(ylabel, fontsize = 24)
            plt.xlabel('Polynom Order', fontsize = 24)
            plt.legend(loc='best' , fontsize = 24)
            plt.savefig(fname=filename +str(i)+'_kf' +title[-2:], dpi='figure', format= 'pdf')
    plt.close('all')

def plotting_r2(toi, filename):
    max_order = toi['Complexity'].max()
    g = sns. FacetGrid(toi, row ='lambda', col='kFold', hue ='Regression type', margin_titles =True)
    g.map(plt.plot, 'Complexity', 'Value')
    g.set_axis_labels('Polynom Order', 'R2')
    g.add_legend()
    g.set(xticks = np.arange(0, max_order +1, 2))
    g.savefig(filename +'.pdf')
    labels = toi['Metric'].unique()
    axes = g.axes.flat
    labels = ['RIDGE', 'LASSO']
    for i, ax in enumerate(axes):
        xax = ax.get_xlim()
        x,y  = duplicate_axis(ax)
        plt.figure(figsize=(10,10))
        if x.ndim == 1:
            plt.plot(x,y)
        else:
            for k in range(len(x)):

                plt.plot(x[k], y[k], label = labels[k])
            plt.legend(loc='best' , fontsize = 24)

        plt.xlim(xax)
        plt.ylabel('R2', fontsize = 24)
        plt.xlabel('Polynom Order', fontsize = 24)
        plt.savefig(fname =filename +str(i), dpi='figure', format= 'pdf')
        plt.close('all')

def plotting(toi, folder = ''):
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
    plotting_mse(toi[lam_filter], row ='Regression type', col='kFold', filename = folder +'reg_types')
    #compare kfold and lambda for lasso
    plotting_mse(toi[lasso_filter], row ='lambda', col='kFold', filename = folder +'lasso_lam_vs_kfold')
    #compare kfold and lambda fore ridge
    plotting_mse(toi[ridge_filter], row ='lambda', col='kFold', filename = folder + 'ridge_lam_vs_kfold')
    #make plot from book
    plotting_mse(toi[book_filter], row ='Regression type', col='kFold', filename=folder +'train_vs_test', split = True, shary=True)
    plotting_mse(toi[book_filter], row ='Regression type', col='kFold', filename=folder +'train_vs_test_no_share', split = True, shary=False)
    #make r2 plot and split (no shared y)
    plotting_r2(toi[r2_filter], filename = folder +'r2')

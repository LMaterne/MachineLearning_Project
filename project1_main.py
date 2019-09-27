import Poly2DFit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional_functions_project1 import benchmarking, toi_append, plotting, load_terrain, terrain
import time

SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def main():

    ###########################################################################################################################
    #######nr of datapoints ###################################################################################################
    ###########################################################################################################################

    n = [10**i for i in range(2,6)]
    reg_types = np.array(['OLS', 'RIDGE', 'LASSO'])
    lam = 10**(-4)
    order = 4

    times = np.zeros((3, len(n)))
    mse = np.zeros((3, len(n)))
    
    fit = Poly2DFit.Poly2DFit()
    
    for i, reg in enumerate(reg_types):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            s = time.perf_counter()
            fit.kfold_cross(order, reg, lam, k=5)
            times[i,j] = time.perf_counter() - s
            #fit.evaluate_model()
            mse[i,j] = fit.mse_train
    

    plt.figure(figsize=(10,10))
    for i, reg in enumerate(reg_types):
        plt.plot(n, times[i], label = reg)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize = 28)
    plt.xlabel('# samples', fontsize = 28)
    plt.ylabel('Elapsed time in s', fontsize = 28)
    plt.xticks(n)
    plt.savefig('time.pdf')

    plt.figure(figsize=(10,10))
    for i, reg in enumerate(reg_types):
        plt.plot(n, mse[i], label = reg)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best', fontsize = 28)
    plt.xlabel('# samples', fontsize = 28)
    plt.ylabel('MSE', fontsize = 28)
    plt.xticks(n)
    plt.savefig('mse.pdf')
    
    num=[25, 100, 400]
    betas = np.zeros((3,2,100))
    for nu in num:
        fit = Poly2DFit.Poly2DFit()
        fit.generateSample(nu) 
        fit.kfold_cross(order, 'OLS', k=5)
        #fit.run_fit(order, 'OLS')
        #fit.evaluate_model()
        par = fit.par
        var = fit.par_var
        par_up = np.linalg.norm(par +  	1.96*np.sqrt(var/nu))
        par_down = np.linalg.norm(par -  	1.96*np.sqrt(var/nu))
        betas[0,0] = par_down * np.ones(100)
        betas[0,1] = par_up * np.ones(100)
        #mse_ols_t = [fit.mse_train for i in range(20)]
        mse_ols = [fit.mse_train for i in range(100)]
        mse_lambda = np.zeros((2,2,100))
        
        lambdas = np.linspace(10**(-10), 10**(-1),100)
        for i, reg in enumerate(reg_types[1:]):
            for j, lam in enumerate(lambdas):  
                        
                fit.kfold_cross(order, reg, lam, k=5)
                #fit.run_fit(order, reg, lam)
                #fit.evaluate_model()
                par = fit.par
                var = fit.par_var
                if(i == 0):
                    par_up = np.linalg.norm(par +  	1.96*np.sqrt(var/nu))
                    par_down = np.linalg.norm(par -  	1.96*np.sqrt(var/nu))
                    betas[i+1,0,j] = par_down 
                    betas[i+1,1,j] = par_up 
                else:
                    betas[i+1,0,j] = np.linalg.norm(par) 
                mse_lambda[i,0,j] = fit.mse_train
                mse_lambda[i,1,j] = fit.mse_train
        
        c = [ 'tab:blue', 'tab:orange', 'tab:green']
        plt.figure(figsize=(10,10))
        plt.fill_between(lambdas,betas[0,0], betas[0,1], label ='OLS', alpha = 0.5, color = c[2], edgecolor = c[2] )
        plt.fill_between(lambdas,betas[1,0], betas[1,1], label ='RIDGE', alpha = 0.5, color = c[0], edgecolor = c[0])
        plt.plot(lambdas, betas[2,0], label = 'LASSO', color = c[1])
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim(top=350)
        plt.legend(loc='best', fontsize = 28)
        plt.xlabel('$\lambda$', fontsize = 28)
        plt.ylabel(r'|$\hat{\beta}$|', fontsize = 28)
        plt.savefig('lambda_par_%i_p4.pdf'%nu)
        
        plt.figure(figsize=(10,10))
        plt.plot(lambdas, mse_ols,color = c[2], label = 'OLS')
        #plt.plot(lambdas, mse_ols_t,color = c[2],linestyle ='dashed', label = 'OLS Test')
        for i, reg in enumerate(reg_types[1:]):
            plt.plot(lambdas, mse_lambda[i,0], color = c[i], label = reg)
            #plt.plot(lambdas, mse_lambda[i,0], color = c[i],linestyle ='dashed', label = reg + ' Test')
        plt.xscale('log')
        plt.legend(loc='best', fontsize = 28)
        plt.xlabel('$\lambda$', fontsize = 28)
        plt.ylabel('MSE', fontsize = 28)
        plt.ylim(0.25,1.4)
        plt.savefig('lambda_%i_p4.pdf'%nu)
    
    order = [6,9,12,15]
    mse_ord = np.zeros((len(order),len(n)))
    for i, p in enumerate(order):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            fit.kfold_cross(p, 'OLS', lam, k=5)
            #fit.evaluate_model()
            mse_ord[i,j] = fit.mse

    plt.figure(figsize=(10,10))
    for i, p in enumerate(order):
        plt.plot(n, mse_ord[i], label = 'Pol. order ' +str(p))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 600)
    plt.legend(loc='best', fontsize = 28)
    plt.xlabel('# samples', fontsize = 28)
    plt.ylabel('MSE', fontsize = 28)
    plt.xticks(n)
    plt.savefig('mse_pol_ols.pdf')

    for i, p in enumerate(order):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            fit.kfold_cross(p, 'RIDGE', lam, k=5)
            #fit.evaluate_model()
            mse_ord[i,j] = fit.mse

    plt.figure(figsize=(10,10))
    for i, p in enumerate(order):
        plt.plot(n, mse_ord[i], label = 'Pol. order ' +str(p))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 600)
    plt.legend(loc='best', fontsize = 28)
    plt.xlabel('# samples', fontsize = 28)
    plt.ylabel('MSE', fontsize = 28)
    plt.xticks(n)
    plt.savefig('mse_pol_ridge.pdf')
    
"""
    ###########################################################################################################################
    #######Benchmarking #######################################################################################################
    ###########################################################################################################################

    ks = [0,2,5,10]
    lam = [10**(-4), 10**(-3),10**(-2)]
    
    max_order = 10
    samples = 10**3

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
    # plot results of benchmarking       
    plotting(toi, folder='./Benchmark/')
    ###########################################################################################################################
    #######Terrain Data #######################################################################################################
    ###########################################################################################################################
    terrain_dat = load_terrain('./Terraindata/yellowstone2')
    xl, yl = terrain_dat.shape
    
    #quater of array
    xl = xl//2
    yl = yl//2
    print(xl,yl)
    c = plt.imshow(terrain_dat[:xl,:yl], cmap='gray')
    plt.colorbar(c, label = 'Hight')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./Terraindata/yellowstone2_section')
    plt.close('all')
    points = np.zeros((xl*yl, 3))
    #pixelpositions or better linspace?
    x_ax = np.arange(0,xl)
    y_ax = np.arange(0,yl)
    #flatten 2D array
    count = 0
    for x in x_ax:
        for y in y_ax:
            points[count] = [x, y, terrain_dat[x,y]]
            count += 1
    x,y,z = points.T

    max_order = 6
    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    ops = len(ks) * ( 2 * len(lam) + 1)
    one_part = 100 / ops #in%
    current_progress = 0
    print()
    print('Terrain data')
    for k in ks:
        current_progress += one_part
        print("Now: OLS; Progress", int(current_progress), "%")
        temp = terrain('OLS', x,y,z, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        
        for l in lam:
            current_progress += one_part
            print("Now: RIDGE; Progress", int(current_progress), "%")
            temp = terrain('RIDGE', x,y,z, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)


            current_progress += one_part
            print("Now: LASSO; Progress", int(current_progress), "%")      
            temp = terrain('LASSO', x,y,z, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'LASSO', l, k)     

    # plot results of terrain fit       
    plotting(toi, folder='./Terrain_yellowstone2_section/')
    
    terrain_fit = Poly2DFit.Poly2DFit()
    terrain_fit.givenData(x, y, z)
    par,_ = terrain_fit.run_fit(8, 'OLS', 0.01)
    X = terrain_fit._design
    fit_map = X.dot(par).reshape(xl,yl)
    
    c = plt.imshow(fit_map, cmap='gray')
    plt.colorbar(c, label = 'Hight')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./Terraindata_yellostone2_section/yellowstone2_OLS.pdf')

    ###########################################################################################################################
    #######Terrain Data #######################################################################################################
    ###########################################################################################################################
    terrain_dat = load_terrain('./Terraindata/yellowstone2')
    terrain_dat = terrain_dat[::4,::4]
    xl, yl = terrain_dat.shape
    print(xl,yl)
    #quater of array
    #xl = xl//2
    #yl = yl//2

    c = plt.imshow(terrain_dat[:xl,:yl], cmap='gray')
    plt.colorbar(c, label = 'Hight')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./Terrain_yellowstone2_low/yellowstone2_low_res')
    plt.close('all')
    points = np.zeros((xl*yl, 3))
    #pixelpositions or better linspace?
    x_ax = np.arange(0,xl)
    y_ax = np.arange(0,yl)
    #flatten 2D array
    count = 0
    for x in x_ax:
        for y in y_ax:
            points[count] = [x, y, terrain_dat[x,y]]
            count += 1
    x,y,z = points.T

    max_order = 8
    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )
    ops = len(ks) * ( 2 * len(lam) + 1)
    one_part = 100 / ops #in%
    current_progress = 0
    print()
    print('Terrain data')
    for k in ks:
        current_progress += one_part
        print("Now: OLS; Progress", int(current_progress), "%")
        temp = terrain('OLS', x,y,z, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        
        for l in lam:
            current_progress += one_part
            print("Now: RIDGE; Progress", int(current_progress), "%")
            temp = terrain('RIDGE', x,y,z, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)


            current_progress += one_part
            print("Now: LASSO; Progress", int(current_progress), "%")      
            temp = terrain('LASSO', x,y,z, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'LASSO', l, k)     

    # plot results of terrain fit       
    plotting(toi, folder='./Terrain_yellowstone2_low/')
    
    terrain_fit = Poly2DFit.Poly2DFit()
    terrain_fit.givenData(x, y, z)
    par,_ = terrain_fit.run_fit(8, 'OLS', 0.01)
    X = terrain_fit._design
    fit_map = X.dot(par).reshape(xl,yl)
    
    c = plt.imshow(fit_map, cmap='gray')
    plt.colorbar(c, label = 'Hight')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./Terrain_yellowstone2_low/yellowstone2_OLS.pdf')
"""


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")
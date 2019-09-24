import Poly2DFit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional_functions_project1 import benchmarking, toi_append, plotting, load_terrain, terrain
import time

def main():
    """
    ###########################################################################################################################
    #######nr of datapoints ###################################################################################################
    ###########################################################################################################################

    n = [10**i for i in range(1,7)]
    reg_types = ['OLS', 'RIDGE', 'LASSO']
    lam = 10**(-2)
    order = 4

    times = np.zeros((3, len(n)))
    mse = np.zeros((3, len(n)))

    fit = Poly2DFit.Poly2DFit()
    for i, reg in enumerate(reg_types):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            s = time.perf_counter()
            fit.run_fit(order, reg, lam)
            times[i,j] = time.perf_counter() - s
            fit.evaluate_model()
            mse[i,j] = fit.mse

    plt.figure(figsize=(10,10))
    for i, reg in enumerate(reg_types):
        plt.plot(n, times[i], label = reg)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', fontsize = 24)
    plt.xlabel('# samples', fontsize = 24)
    plt.ylabel('Elapsed time in s', fontsize = 24)
    plt.xticks(n)
    plt.savefig('time.pdf')

    plt.figure(figsize=(10,10))
    for i, reg in enumerate(reg_types):
        plt.plot(n, mse[i], label = reg)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best', fontsize = 24)
    plt.xlabel('# samples', fontsize = 24)
    plt.ylabel('log(MSE)', fontsize = 24)
    plt.xticks(n)
    plt.savefig('mse.pdf')

    order = [6,9,12,15]
    mse_ord = np.zeros((len(order),len(n)))
    for i, p in enumerate(order):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            fit.run_fit(p, 'OLS', lam)
            fit.evaluate_model()
            mse_ord[i,j] = fit.mse

    plt.figure(figsize=(10,10))
    for i, p in enumerate(order):
        plt.plot(n, mse_ord[i], label = 'Pol. order ' +str(p))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 600)
    plt.legend(loc='best', fontsize = 24)
    plt.xlabel('# samples', fontsize = 24)
    plt.ylabel('log(MSE)', fontsize = 24)
    plt.xticks(n)
    plt.savefig('mse_pol_ols.pdf')

    for i, p in enumerate(order):
        for j, samples in enumerate(n):
            fit.generateSample(samples)
            fit.run_fit(p, 'RIDGE', lam)
            fit.evaluate_model()
            mse_ord[i,j] = fit.mse

    plt.figure(figsize=(10,10))
    for i, p in enumerate(order):
        plt.plot(n, mse_ord[i], label = 'Pol. order ' +str(p))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 600)
    plt.legend(loc='best', fontsize = 24)
    plt.xlabel('# samples', fontsize = 24)
    plt.ylabel('log(MSE)', fontsize = 24)
    plt.xticks(n)
    plt.savefig('mse_pol_ridge.pdf')
    
    """
    ###########################################################################################################################
    #######Benchmarking #######################################################################################################
    ###########################################################################################################################

    ks = [0,2,4,7,10]
    lam = [10**(-5), 10**(-4),10**(-3), 10**(-2)]
    """
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
    """
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






if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")
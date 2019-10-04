import Poly2DFit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional_functions_project1 import benchmarking, toi_append, load_terrain
from plotting_functions import plotting
from imageio import imread
import time
def normalize(x,y,z, rescale = True):
    """
    normalize x,y, z 
    if rescale = True -> shift mean(z) -> 0
    """
    x = x / x.max()
    y = y / y.max()
    z =  z / z.max()
    if rescale:
        z -= np.mean(z)
    return x,y,z

def main():
    
    ks = [0, 5]
    lam = [10**(-5),10**(-3),10**(-1)]

    max_order = 12
    
    samples = 5*10**2

    toi = pd.DataFrame(columns = ['Regression type','lambda','kFold',
                                        'Complexity','Value', 'Metric'] )

    ops = len(ks) * ( 2 * len(lam) + 1)
    one_part = 100 / ops #in%
    current_progress = 0

# load terrain data, for example yellowstone
#    x,y,z = load_terrain('yellowstone')
#Give x, y,z as variables to benchmarking when working with terrain data.

    for k in ks:
        current_progress += one_part
        print("Now: OLS; Progress", int(current_progress), "%")
        temp,coef = benchmarking('OLS', samples, max_order+1, kfold=k, plot_info= False, display_info = False)
        toi = toi_append(toi, temp, 'OLS', 0, k)
        coef.to_csv('./Benchmark/olsbeta.csv')
        for l in lam:
            current_progress += one_part
            print("Now: RIDGE; Progress", int(current_progress), "%")
            temp,coef = benchmarking('RIDGE', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'RIDGE', l, k)
            coef.to_csv('./Benchmark/ridgebeta.csv')

            current_progress += one_part
            print("Now: LASSO; Progress", int(current_progress), "%")

            temp,coef = benchmarking('LASSO', samples, max_order+1, lam=l, kfold= k, plot_info= False, display_info = False)
            toi = toi_append(toi, temp, 'LASSO', l, k)
            coef.to_csv('./Benchmark/lassobeta.csv')
    
    # plot results of benchmarking
    toi.to_csv('./Benchmark/benchmarking.csv')
    plotting(toi, folder='./Benchmark/')
    


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")

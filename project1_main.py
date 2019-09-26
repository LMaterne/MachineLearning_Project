import Poly2DFit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional_functions_project1 import benchmarking, toi_append, load_terrain
from plotting_functions import plotting
import time

def main():

    ks = [0, 1, 5, 10]
    lam = [10**(-5), 10**(-3), 10**(-1)]

    max_order = 6
    samples = 5*10**3

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
    plotting(toi, folder='')


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    finish = time.perf_counter()
    print("Elapsed time: ", finish - start, "s")

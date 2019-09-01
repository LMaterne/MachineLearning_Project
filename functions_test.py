from functions_project1 import matDesign, linReg, MSE, R2
import unittest
from numpy.random import randn, rand
import numpy as np 

np.random.seed(101)

class TestMethodes(unittest.TestCase):
    """
    testing critical functions of functions.py
    """
    def test_linReg(self):
        x,y = rand(2,100)
        design = matDesign([x,y], 2, 2)
        ideal_par = np.array([4, 5, -3, 1, 2, -1] )
        data = design.dot(ideal_par) + randn(100)
        fit_res = linReg(data, design)
        res = np.abs(fit_res - ideal_par)
        self.assertAlmostEqual(res.max(),0)
    
    def test_MSE(self):
        x = rand(10000)
        y = 5*x +4
        data = 5*x +4 + randn(10000)
        mse = MSE(data, y)
        self.assertAlmostEqual(mse,1)
    def test_MSE_as_Var(self):
        data = randn(10000)
        mse = MSE(data, data.mean())
        self.assertAlmostEqual(mse,1)
    

if __name__ == '__main__':
    unittest.main()
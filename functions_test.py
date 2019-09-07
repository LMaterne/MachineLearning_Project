import Poly2DFit 
import numpy as np 
import pytest

"""
run pytest command from comandline 
automatically executes all test_*.py or *_test.py files
"""

np.random.seed(2019)

def test_Poly2DFit():
    """
    testing the functionallity of Poly2DFit by try to reconstruct a 
    given parameter example for a polynomial of degree 3 i.e 10 random parameters
    for x,y in [-1, 1]
    """
    par  = 20*np.random.randn(10) + 2
    x,y = 2 * np.random.rand(2,100)  - 1 
    create_fit = Poly2DFit.Poly2DFit()
    #asign data 
    create_fit.x = x
    create_fit.y = y
    create_fit.order = 3
    design = create_fit.matDesign()
    #assigning data to fit to with X.beta
    data = design.dot(par)
    #free memory and test with new class instance
    del create_fit 
    test_fit = Poly2DFit.Poly2DFit()
    test_fit.givenData(x, y, data)

    #test OLS###########################################################################
    test_par, _ = test_fit.run_fit(3, 'OLS')
    res = np.abs(test_par -par).max()
    #test assertion to precison of 10^-9
    assert res == pytest.approx( 0 ,  abs = 1e-9 ) 
    
    #test RIDGE#########################################################################
    #only accurate for lam = 0 ?
    test_par, _ = test_fit.run_fit(3, 'RIDGE', 0)
    res = np.abs(test_par -par).max()
    #test assertion to precison of 10^-9
    assert res == pytest.approx( 0 ,  abs = 1e-9 ) 

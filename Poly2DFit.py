from additional_functions_project1 import MSE, R2, FrankeFunction
import numpy as np 

  
class Poly2DFit:
    """
    class which perfomers a 2D polynomial fit to givven data or generatd samples from the Franke function
    Class Variables:
     -dependentvariables are stored in x,y and the constructed design matrix in _design
     -data to fit in data
     -order of the polynomial to use in order
     -all perfomrance information are stored in mse, r2, variance, bias
     - parameters and theri variance are stored in par, par_var
    Class Methodes:
        -generateSample from Franke function
        -givenData input real data
        -matDesign creates a design matrix
        -_linReg and _ridgeReg calculate parameters with respectivly regression type and calculate parameter variance
        - runFit performes the fit
    """

    def generateSample(self, n, mean = 0, var = 1):
        """
        Generates (n,3) samples [x,y,z], where x,y are uniform random numbers [0,1) 
        and z = f(x,y) + eps with f the Franke function and eps normal distributed with mean and var
        """
        self.x, self.y = np.random.rand(2,n)
        self.data = FrankeFunction(self.x, self.y) + np.sqrt(var)*np.random.randn(n) + mean

    def givenData(self, x, y, f):
        """
        stores given 2D data in class
        x,y are dependent variables, f= f(x,y)
        """        
        self.x = x
        self.y = y
        self.data = f

    def matDesign (self, indVariables = 2):
    '''This is a function to set up the design matrix
    the inputs are :dataSet, the n datapoints, x and y data in a nx2 matrix
                    order, is the order of the coefficients, 
                    indVariables, the number of independant variables
                    
    i.e if order = 3 and indVariables = 1, then the number of coefficients THIS function will create is 4. (1 x x**2 x**3)
    or  if order = 2 and indVariables = 2, then the number of coefficients THIS function will create is 6. (1 x y xy x**2 y**2) 
    
    IMPORTANT NOTE: this works only for indVariables = 2 at the moment
    
    the outputs are X
    '''
    #stack data
    dataSet = np.vstack(self.x, self.y).T

    # if statement for the case with one independant variable
    if indVariables == 1:
        num_coeff = int(self.order + 1)
        
        # set up the Design matrix
        #n = np.int(np.size(dataSet))
        #matX = np.zeros((n,coefficients))
        
        n = np.shape(dataSet)[0]
        # loop through all the other columns as powes of dataSet
        
        self._design = np.zeros((n,num_coeff))
        i = 0 #counter
        while i < num_coeff:

            self.design[:,i] = (dataSet[i])**i
            i=i+1
        
        
    ###########################################################################################################
    
    # if statement for the case with two independant variables
    
    if (indVariables == 2):
        # find the number of coefficients we will end up with
        num_coeff = int((self.order + 1)*(self.order + 2)/2)
        #print ('The number of coefficients are: ',num_coeff)
                
        #find the number of rows in dataSet
        n = np.shape(dataSet)[0]
        #print ('The number of rows in the design matrix is', n)
        # create an empty matrix of zeros
        self._design = np.zeros((n,num_coeff))
        

        
        col_G = 0 # global columns        
        tot_rows = n
        #print ('total rows = ',tot_rows)
        
        j = 0        
        # loop through each j e.g 1,2,3,4,5,6
        while j < num_coeff:
            k = 0
            #loop through each row
            while k <= j:                   
                row = 0                
                #loop through each item (each column in the row)
                while row < tot_rows:
                    self._design[row,col_G] = ((dataSet[row,0])**(j-k)) * ((dataSet[row,1])**k)                                        
                    row = row + 1 
                    #print(row)
                    
                k = k + 1                     
            col_G = col_G + 1            
            j = j + 1 

    def _linReg(self):
        """
        calculates the estimated parameters of an OLS
        outputs variance as the diagonal entries of (X^TX)^-1
        """
        inverse = np.linalg.inv(self._design.T.dot(self._design))
        self.par_var = np.diag(inverse)
        self.par = inverse.dot(self._design.T).dot(self.data)

    def _ridgeReg(self):
        """
        returns the estimated parameters of an Ridge Regression with 
        regularization parameter lambda
        outputs variance as the diagonal entries of (X^TX- lam I)^-1
        """
        #creating identity matrix weighted with lam
        diag = self.lam * np.ones(design.shape[1])
        inverse = np.linalg.inv(self._design.T.dot(self._design) + np.diag(diag))
        self.par_var = np.diag(inverse)
        self.par = inverse.dot(self._design.T).dot(self.data)
        
    
    def run_fit(self, Pol_order, regtype, lam = 0.1):
        """
        perfomes the fit of the data to the model given as design matrix
        suportet regtypes are 'OLS', 'RIDGE'
        lam is ignored for OLS
        returns fit parameters and their variance
        """
        self.order = Pol_order
        self.lam = lam
        Poly2DFit.matDesign(self)

        if regtype == 'OLS':
            _linReg(data, design)
        if regtype == 'RIDGE':
            _ridgeReg(data, design, lam)

        return self.par, self.par_var

    def evaluate_model(self):
        """
        -calculates the MSE
        -calcualtes the variance and bias of the modell
        returns the modelpoints
        """
        p = self.par.shape[0]

        model = self._design.dot(self.par)
        expect_model = np.mean(model)

        self.mse = MSE(self.data, model)
        self.r2 = R2(data, model)

        self.bias = MSE( self.data, expect_model)
        self.variance = MSE(model, expect_model)
        return self.x, self.y, model
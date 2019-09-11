from additional_functions_project1 import MSE, R2, FrankeFunction, plot_it, kfold
import numpy as np 
import subprocess
import warnings


  
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
        np.random.seed(0)
        self.x, self.y = np.random.rand(2,n)
        self.data = FrankeFunction(self.x, self.y) + np.sqrt(var)*np.random.randn(n) + mean

    def generateKfold(self, n, mean = 0, var = 1):
        """
        This is a function to generate [x,y,z] using kfold cross validation.
        The function generates uniform random numbers [0,1] then performs n fold resamples.
        i.e it splits a sample into training and test data and finds the mean of the training data.
        It the repeats this n times
        
        """
        np.random.seed(0)
        x, y = np.random.rand(2,n)
        self.x, self.y = kfold(x,y)
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
        dataSet = np.vstack((self.x, self.y)).T

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

                self._design[:,i] = (dataSet[i])**i
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
        XTX = self._design.T.dot(self._design)
        #try to use standard inversion, otherwise use SVD
        try:
            inverse = np.linalg.inv(XTX)
        except:
            print("in exception")
            raise warnings.warn("Singular Matrix: Using SVD", Warning)
            U, S, VT = np.linalg.svd(XTX)
            inverse = VT.T.dot(np.diag(1/S)).dot(U.T)

        self.par_var = np.diag(inverse)
        self.par = inverse.dot(self._design.T).dot(self.data)

    def _ridgeReg(self):
        """
        returns the estimated parameters of an Ridge Regression with 
        regularization parameter lambda
        outputs variance as the diagonal entries of (X^TX- lam I)^-1
        """
        #creating identity matrix weighted with lam
        diag = self.lam * np.ones(self._design.shape[1])
        XTX_lam = self._design.T.dot(self._design) + np.diag(diag)

        #try to use standard inversion, otherwise use SVD
        try:
            inverse = np.linalg.inv(XTX_lam)
        except:
            warnings.warn("Singular Matrix: Using SVD", Warning)
            U, S, VT = np.linalg.svd(XTX_lam)
            inverse = VT.T.dot(np.diag(1/S)).dot(U.T)

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
        self.regType = regtype
        Poly2DFit.matDesign(self)

        if regtype == 'OLS':
            Poly2DFit._linReg(self)
        if regtype == 'RIDGE':
            Poly2DFit._ridgeReg(self)

        return self.par, self.par_var

    def evaluate_model(self):
        """
        -calculates the MSE
        -calcualtes the variance and bias of the modell
        returns the modelpoints
        """
        p = self.par.shape[0]

        self.model = self._design.dot(self.par)
        expect_model = np.mean(self.model)

        self.mse = MSE(self.data, self.model)
        self.r2 = R2(self.data, self.model)
      
        self.bias = MSE(FrankeFunction(self.x, self.y), expect_model) # explain this in text why we use FrankeFunction
        self.variance = MSE(self.model, expect_model)
        return self.x, self.y, self.model
    
   
    def plot_function(self):
        
        self.plot_function = plot_it(self.x,self.y, self.model, self.data)
         

    def store_information(self, filepath, filename):
    
        try:
            f = open(filepath + "/" + filename  + ".txt",'w+')
        except:
            subprocess.call(["mkdir", "-p", filepath ])
            f = open(filepath + "/"+ filename + ".txt",'w+')

        f.write("    Perfomance of %s regression with  %i parameters \n:" %(self.regType, len(self.par)))
        
        if self.regType != 'OLS':
            f.write("Regularization parameter lambda = %f\n" %self.lam)
        
        f.write("MSE = %.4f \t R2 = %.4f \t Bias(model)=%.4f \t Variance(model) =%.4f \n" %(self.mse, self.r2, self.bias, self.variance))
        f.write("Parameter Information:\n")
        for i in range(len(self.par)):
            f.write("beta_%i = %.4f +- %.4f\n" %(i, self.par[i], np.sqrt(self.par_var[i])) )
        f.close()
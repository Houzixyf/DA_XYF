import numpy as np
from numpy.linalg import solve, norm
import scipy as scp
import time

from log import logging



class Solver:
    '''
    This class provides solver for the collocation equation system.
    
    
    Parameters
    ----------
    
    F : callable
        The callable function that represents the equation system
    
    DF : callable
        The function for the jacobian matrix of the eqs
    
    x0: numpy.ndarray
        The start value for the solver
    
    tol : float
        The (absolute) tolerance of the solver
    
    maxIt : int
        The maximum number of iterations of the solver
    
    method : str
        The solver to use
    par : np.array
    '''
    # Solver(F=G, DF=DG, x0=self.guess,
    def __init__(self, F, DF, x0, tol=1e-5, reltol=2e-5, maxIt=100, method='leven', par_k = np.array([0.0])):
        # x0 = free_param
        self.F = F
        self.DF = DF
        self.x0 = x0 ##:: x0=self.guess initial: array([ 0.1,  0.1,  0.1, ...,  0.1,  0.1,  z0])
        self.tol = tol
        self.reltol = reltol
        self.maxIt = maxIt
        self.method = method
        # self.itemindex = itemindex
        self.sol = None
        self.par = par_k
    

    def solve(self):
        '''
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        '''
        
        if (self.method == 'leven'):
            logging.debug("Run Levenberg-Marquardt method")
            self.leven()
        
        if (self.sol is None):
            logging.warning("Wrong solver, returning initial value.")
            return self.x0
        else:
            return self.sol, self.par


    def leven(self):
        '''
        This method is an implementation of the Levenberg-Marquardt-Method
        to solve nonlinear least squares problems.
        
        For more information see: :ref:`levenberg_marquardt`
        '''
        i = 0
        x = self.x0 ##:: guess_value
        res = 1 ##:: residuum
        res_alt = -1
        
        eye = scp.sparse.identity(len(self.x0)) ##:: diagonal matrix, value: 1.0, danwei

        #mu = 1.0
        mu = 1e-4
       # mu_old = mu
        
        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        roh = 0.0

        reltol = self.reltol # reltol=2e-5
        
        Fx = self.F(x) # G, self.F: method, self.F(x): calling function
        
        # measure the time for the LM-Algorithm
        T_start = time.time()
        
        while((res > self.tol) and (self.maxIt > i) and (abs(res-res_alt) > reltol)):
            i += 1
            
            #if (i-1)%4 == 0:
            DFx = self.DF(x) ##:: part of Jacobi-Matrix J
            DFx = scp.sparse.csr_matrix(DFx)
            
            break_inner_loop = False
            while (not break_inner_loop):                
                A = DFx.T.dot(DFx) + mu**2*eye ##:: left side of equation, J'J+mu^2*I, Matrix.T=inv(Matrix)

                b = DFx.T.dot(Fx) ##:: right side of equation, J'f, (f=Fx)
                    
                s = -scp.sparse.linalg.spsolve(A,b) ##:: h

                xs = x + np.array(s).flatten()
                
                Fxs = self.F(xs)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                R1 = (normFx**2 - normFxs**2) ##:: F(x)^2-F(x+h)^2, F(x)=f
                R2 = (normFx**2 - (norm(Fx+DFx.dot(s)))**2) # F(x)^2-(F(x)+F'(x)h)^2
                
                R1 = (normFx - normFxs)
                R2 = (normFx - (norm(Fx+DFx.dot(s))))
                roh = R1 / R2
                
                # note smaller bigger mu means less progress but
                # "more regular" conditions
                
                if R1 < 0 or R2 < 0:
                    # the step was too big -> residuum would be increasing
                    mu*= 2
                    roh = 0.0 # ensure another iteration
                    
                    #logging.debug("increasing res. R1=%f, R2=%f, dismiss solution" % (R1, R2))

                elif (roh<=b0):
                    mu = 2*mu
                elif (roh>=b1):
                    
                    mu = 0.5*mu

                # -> if b0 < roh < b1 : leave mu unchanged
                
                logging.debug("  roh= %f    mu= %f"%(roh,mu))
                
                if roh < 0:
                    logging.warn("roh < 0 (should not happen)")
                
                # if the system more or less behaves linearly 
                break_inner_loop = roh > b0
            
            Fx = Fxs # F(x+h) -> Fx_new
            x = xs # x+h -> x_new
            
            #roh = 0.0
            res_alt = res
            res = normFx
            if i>1 and res > res_alt:
                logging.warn("res_old > res  (should not happen)")

            logging.debug("nIt= %d    res= %f"%(i,res))

        # LM Algorithm finished
        T_LM = time.time() - T_start
        self.avg_LM_time = T_LM / i
        
        self.sol = x # return (x+h)
        self.par = np.array([self.sol[-1]]) # self.itemindex

    # def call_par(self):
    #     return self.par
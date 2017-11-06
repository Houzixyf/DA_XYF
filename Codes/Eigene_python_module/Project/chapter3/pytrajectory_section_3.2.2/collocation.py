# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
from scipy import linalg

from log import logging, Timer
from trajectories import Trajectory
from solver import Solver

from auxiliary import sym2num_vectorfield


np.set_printoptions(threshold='nan')    
 

# from IPython import embed as IPS

class Container(object):
    """
    Simple data structure to store additional internal information for
    debugging and checking the algorithms.
    Some of the attributes might indeed be neccessary
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.__setattr__(str(key), value)


class CollocationSystem(object):
    '''
    This class represents the collocation system that is used
    to determine a solution for the free parameters of the
    control system, i.e. the independent variables of the
    trajectory splines.
    
    Parameters
    ----------
    
    sys : system.DynamicalSystem
        Instance of a dynamical system
    '''

    def __init__(self, sys, constraints, **kwargs):
        self.sys = sys # sys=self.dyn_sys ##:: yk
        
        # set parameters
        self._parameters = dict()
        self._parameters['tol'] = kwargs.get('tol', 1e-5) ##:: Tolerance
        self._parameters['reltol'] = kwargs.get('reltol', 2e-5)
        self._parameters['sol_steps'] = kwargs.get('sol_steps', 100)
        self._parameters['method'] = kwargs.get('method', 'leven')
        self._parameters['coll_type'] = kwargs.get('coll_type', 'equidistant')
        self._parameters['z_par'] = kwargs.get('par', [1.0])
        self.constraints = constraints
        ##!! self.n_par = self._parameters['z_par'].__len__()
        # we don't have a soution, yet
        self.sol = None
        
        # create vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())
        f = sys.f_sym(sp.symbols(sys.states), sp.symbols(sys.inputs), sp.symbols(sys.par) ) ##:: f_sym is a function, but here the self-variable are already input, so f is value, not function. f = array([x2, u1, x4, -u1*(0.9*cos(x3) + 1) - 0.9*x2**2*sin(x3)])
        
        # TODO_ok: check order of variables of differentiation ([x,u] vs. [u,x])
        #       because in dot products in later evaluation of `DG` with vector `c`
        #       values for u come first in `c`
        
        # TODO_ok: remove this comment after reviewing the problem
        # previously the jacobian was calculated wrt to strings which triggered strange
        # strange sympy behavior (bug) for systems with more than 9 variables
        # workarround: we use real symbols now
        all_symbols = sp.symbols(sys.states + sys.inputs + sys.par) ##:: (x1, x2, x3, x4, u1)
        Df = sp.Matrix(f).jacobian(all_symbols) ##:: sp.Matrix(f): Matrix([[x2],[u1],[x4],[-u1*(0.9*cos(x3) + 1) - 0.9*x2**2*sin(x3)]])

        self._ff_vectorized = sym2num_vectorfield(f, sys.states, sys.inputs, sys.par, vectorized=True, cse=True)
        self._Df_vectorized = sym2num_vectorfield(Df, sys.states, sys.inputs, sys.par, vectorized=True, cse=True)
        self._f = f
        self._Df = Df
    
        self.trajectories = Trajectory(sys, **kwargs)

        self._first_guess = kwargs.get('first_guess', None)

    def build(self, symbeq = False): # C = self.eqs.build()
                     # self.eqs = CollocationSystem(sys=self.dyn_sys, **kwargs)
        '''
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        '''
        logging.debug("Building Equation System")

        symbeq = self.symbeq
        # make symbols local
        states = self.sys.states ##:: ('x1', 'x2', 'x3', 'x4')
        # determine for each spline the index range of its free vars in the concatenated
        # vector of all free vars
        indic = self._get_index_dict() ##:: {'x1': (0, 17), 'x2': (0, 17), 'u1': (0, 17), 'x3': (17, 26), 'x4': (17, 26)}, from 0th to 16th coeff. belong to chain (x1,x2,x3), from 17th to 25th belong to chain(x3,x4)
        
        # compute dependence matrices
        # Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs, Mp, Mp_abs = self._build_dependence_matrices(indic)
        MC = self._build_dependence_matrices(indic)

        # TODO: self._build_dependence_matrices should already return this container

    # in the later evaluation of the equation system `G` and its jacobian `DG`
        # there will be created the matrices `F` and DF in which every nx rows represent the 
        # evaluation of the control systems vectorfield and its jacobian in a specific collocation
        # point, where nx is the number of state variables
        # 
        # if we make use of the system structure, i.e. the integrator chains, not every
        # equation of the vector field has to be solved and because of that, not every row 
        # of the matrices `F` and `DF` is neccessary
        # 
        # therefore we now create an array with the indices of all rows we need from these matrices
        if self.trajectories._parameters['use_chains']:
            eqind = self.trajectories._eqind
        else:
            eqind = range(len(states))

        # `eqind` now contains the indices of the equations/rows of the vector field
        # that have to be solved
        delta = 2
        n_cpts = self.trajectories.n_parts_x * delta + 1
        # this (-> `take_indices`) will be the array with indices of the rows we need
        # 
        # to get these indices we iterate over all rows and take those whose indices
        # are contained in `eqind` (modulo the number of state variables -> `x_len`)
        # when eqind=[3],that is (x4):
        take_indices = np.tile(eqind, (n_cpts,)) + np.arange(n_cpts).repeat(len(eqind)) * len(states)
        
        # here we determine the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = MC.Mdx[take_indices, :] ##:: in e.g.4: the 3rd,7th,...row, <21x26 sparse matrix>
        # here we compute the jacobian matrix of the system/input splines as they also depend on
        # the free parameters
        DXUP = []
        n_states = self.sys.n_states
        n_inputs = self.sys.n_inputs
        n_par = self.sys.n_par

        for i in xrange(n_cpts):
            DXUP.append(np.vstack(( MC.Mx[n_states * i : n_states * (i+1)].toarray(), MC.Mu[n_inputs * i : n_inputs * (i+1)].toarray(), MC.Mp[n_par * i : n_par * (i+1)].toarray() )))
            
        # XU_old = DXU
        DXUP = np.vstack(DXUP)

        DXUP = sparse.csr_matrix(DXUP)

        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self._ff_vectorized
        Df_vec = self._Df_vectorized

        # transform matrix formats for faster dot products
        if symbeq == True:
            # Sparse Matrix Container:
            SMC = Container()
            # convert all 2d arrays (from MC) to sparse datatypes (to SMC)
            for k, v in MC.__dict__.items():
                # Todo:MC == SMC ??
                # SMC.__dict__[k] = v.tocsr()
                SMC.__dict__[k] = v.toarray()
            SMC.DdX = SMC.Mdx[take_indices, :]

        self.n_cpts = n_cpts
        DdX = DdX.tocsr()

        def get_X_U_P(c, sparse=True):

            if sparse: # for debug
                C = SMC
                
            else: # original codes
                C = MC
            X = C.Mx.dot(c)[:, None] + C.Mx_abs  ##:: X = [S1(t=0), S2(0), S1(0.5), S2(0.5), S1(1), S2(1)]
            U = C.Mu.dot(c)[:, None] + C.Mu_abs  ##:: U = [Su(t=0), Su(0.5), Su(1)]
            P = C.Mp.dot(c)[:, None] + C.Mp_abs  ##:: init: P = [1.0,1.0,1.0]

            X = np.array(X).reshape((n_states, -1),
                                 order='F')  ##:: X = array([[S1(0), S1(0.5), S1(1)],[S2(0),S2(0.5),S2(1)]])
            U = np.array(U).reshape((n_inputs, -1), order='F')
            P = np.array(P).reshape((n_par, -1), order='F') ##:: P = array([[k1,k1,k1],[k2,k2,k2]])

            return X, U, P


        # define the callable functions for the eqs
        
        def G(c, info=False):
            """
            :param c: main argument (free parameters)
            :param info: flag for debug
            :param symbeq: flag for calling this function with symbolic c
                            (for debugging)
            :return:
            """
            ##for debugging symbolic display
            #symbeq = True
            #


            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            sparseflag = self.symbeq ##!! not
            # X, U, P = get_X_U_P(c, sparseflag)

            # TODO_ok: check if both spline approaches result in same values here

            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not be solved

            # this is the regular path
            if self.symbeq:
                # reshape flattened X again to nx times nc Matrix
                # nx: number of states, nc: number of collocation points
                c = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name))
                X, U, P = get_X_U_P(c, sparseflag)
                eq_list = [] # F(w) = 0
                F =  ff_vec(X, U, P).ravel(order='F').take(take_indices, axis=0)[:,None]
                # F = self.sys.f_sym(X,U,P).ravel(order='F').take(take_indices, axis=0)[:,None]
                dX = SMC.Mdx.dot(c)[:,None] + SMC.Mdx_abs
                dX = dX.take(take_indices, axis=0)
                F2 = F - dX
                # the following makes F2 easier to read
                eq_list = F2.reshape(self.n_cpts, self.sys.n_states, -1)         

                resC = Container(X, U, P, G=eq_list)
                return resC
            
            else:
                X, U, P = get_X_U_P(c, sparseflag)
                F = ff_vec(X, U, P).ravel(order='F').take(take_indices, axis=0)[:,None] ##:: F now numeric
                dX = MC.Mdx.dot(c)[:,None] + MC.Mdx_abs
                dX = dX.take(take_indices, axis=0)
                #dX = np.array(dX).reshape((x_len, -1), order='F').take(eqind, axis=0)

                G = F - dX
                res = np.asarray(G).ravel(order='F')
    
                # debug:
                if info:
                    # see Container docstring for motivation
                    iC = Container(X=X, U=U, F=F, P=P, dX=dX, res=res)
                    res = iC
    
                return res

        # and its jacobian
        def DG(c):
            """
            :param c: main argument (free parameters)
            :param symbeq: flag for calling this function with symbolic c
                    (for debugging)
            :return:
            """

            # for debugging symbolic display
            # symbeq = True
            #
            
            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            sparseflag = self.symbeq # not
            X, U, P = get_X_U_P(c, sparseflag)
            
            if self.symbeq:
                c = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name))
                DF_blocks = Df_vec(X,U,P).transpose([2,0,1])
                DF_sym = linalg.block_diag(*DF_blocks).dot(DXUP.toarray()) ##:: array(dtype=object)
                if self.trajectories._parameters['use_chains']:
                    DF_sym = DF_sym.take(take_indices, axis=0)
                DG = DF_sym - SMC.DdX
                
                # the following makes DG easier to read
                DG = DG.reshape(self.n_cpts, self.sys.n_states, -1)

                return DG
                
            else:
                # Todo:
                # get the jacobian blocks and turn them into the right shape
                DF_blocks = Df_vec(X,U,P).transpose([2,0,1])
    
                # build a block diagonal matrix from the blocks
                DF_csr = sparse.block_diag(DF_blocks, format='csr').dot(DXUP)
            
                # if we make use of the system structure
                # we have to select those rows which correspond to the equations
                # that have to be solved
                if self.trajectories._parameters['use_chains']:
                    DF_csr = sparse.csr_matrix(DF_csr.toarray().take(take_indices, axis=0))
                    # TODO: is the performance gain that results from not having to solve
                    #       some equations (use integrator chains) greater than
                    #       the performance loss that results from transfering the
                    #       sparse matrix to a full numpy array and back to a sparse matrix?
            
                DG = DF_csr - DdX
            
                return DG

        C = Container(G=G, DG=DG,
                      Mx=MC.Mx, Mx_abs=MC.Mx_abs,
                      Mu=MC.Mu, Mu_abs=MC.Mu_abs,
                      Mp=MC.Mp, Mp_abs=MC.Mp_abs,
                      Mdx=MC.Mdx, Mdx_abs=MC.Mdx_abs,
                      guess=self.guess)
        
        # return the callable functions
        #return G, DG

        # store internal information for diagnose purposes
        C.take_indices = take_indices
        self.C = C
        return C

    def _get_index_dict(self):
        # here we do something that will be explained after we've done it  ;-)
        indic = dict()
        i = 0
        j = 0
    
        # iterate over spline quantities
        for k, v in sorted(self.trajectories.indep_vars.items(), key=lambda (k, v): k): ##:: items:(key,value)
            # increase j by the number of indep variables on which it depends
            j += len(v)
            indic[k] = (i, j) ##:: indic={'x1':(0,17), 'x3':(17,26)}
            i = j
    
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if self.trajectories._parameters['use_chains']:
            for sq in self.sys.states + self.sys.inputs:
                for ic in self.trajectories._chains:
                    if sq in ic: ##:: if sq is a key of ic
                        indic[sq] = indic[ic.upper]  ##:: {'x1': (0, 17), 'x2': (0, 17), 'u1': (0, 17), 'x3': (17, 26), 'x4': (17, 26)}
    
        # as promised: here comes the explanation
        #
        # now, the dictionary 'indic' looks something like
        #
        # indic = {u1 : (0, 6), x3 : (18, 24), x4 : (24, 30), x1 : (6, 12), x2 : (12, 18)}
        #
        # which means, that in the vector of all independent parameters of all splines
        # the 0th up to the 5th item [remember: Python starts indexing at 0 and leaves out the last]
        # belong to the spline created for u1, the items with indices from 6 to 11 belong to the
        # spline created for x1 and so on...
        
        return indic

    def _build_dependence_matrices(self, indic):
        # first we compute the collocation points
        cpts = collocation_nodes(a=self.sys.a, b=self.sys.b,
                                 npts=self.trajectories.n_parts_x * 2 + 1,
                                 coll_type=self._parameters['coll_type']) ##:: cpts = array([ 0.  ,  0.09,  0.18, ...,  1.62,  1.71,  1.8 ]), n_parts_x mot defined by users in e.g.4, default=10
        x_fnc = self.trajectories.x_fnc ##:: {'x1': methode Spline.f, ...}
        dx_fnc = self.trajectories.dx_fnc
        u_fnc = self.trajectories.u_fnc

        states = self.sys.states
        inputs = self.sys.inputs
        par = self.sys.par
        
        # total number of independent variables
        free_param = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name)) ##:: array([cu1_0_0, cu1_1_0, cu1_2_0, ..., cx4_8_0, cx4_9_0, cx4_0_2, k])
        n_dof = free_param.size
        
        # store internal information:
        self.dbgC = Container(cpts=cpts, indic=indic, dx_fnc=dx_fnc, x_fnc=x_fnc, u_fnc=u_fnc)
        self.dbgC.free_param=free_param

        lx = len(cpts) * self.sys.n_states ##:: number of points * number of states
        lu = len(cpts) * self.sys.n_inputs
        lp = len(cpts) * self.sys.n_par
        
        # initialize sparse dependence matrices
        Mx = sparse.lil_matrix((lx, n_dof))
        Mx_abs = sparse.lil_matrix((lx, 1))
        
        Mdx = sparse.lil_matrix((lx, n_dof))
        Mdx_abs = sparse.lil_matrix((lx, 1))
        
        Mu = sparse.lil_matrix((lu, n_dof))
        Mu_abs = sparse.lil_matrix((lu, 1))
        
        Mp = sparse.lil_matrix((lp, n_dof))
        Mp_abs = sparse.lil_matrix((lp, 1))
        
        for ip, p in enumerate(cpts):
            for ix, xx in enumerate(states):
                # get index range of `xx` in vector of all indep variables
                i,j = indic[xx] ##:: indic = {'x2': (0, 17), 'x3': (17, 26), 'x1': (0, 17), 'u1': (0, 17), 'x4': (17, 26)}

                # determine derivation order according to integrator chains
                dorder_fx = _get_derivation_order(x_fnc[xx])
                dorder_dfx = _get_derivation_order(dx_fnc[xx])
                assert dorder_dfx == dorder_fx + 1

                # get dependence vector for the collocation point and spline variable
                mx, mx_abs = x_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_fx)
                mdx, mdx_abs = dx_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_dfx)

                k = ip * self.sys.n_states + ix
                
                Mx[k, i:j] = mx ##:: Mx.shape = (lx,n_dof)
                Mx_abs[k] = mx_abs

                Mdx[k, i:j] = mdx
                Mdx_abs[k] = mdx_abs
                
            for iu, uu in enumerate(inputs):
                # get index range of `xx` in vector of all indep vars
                i,j = indic[uu]

                dorder_fu = _get_derivation_order(u_fnc[uu])

                # get dependence vector for the collocation point and spline variable
                mu, mu_abs = u_fnc[uu].im_self.get_dependence_vectors(p, d=dorder_fu)

                k = ip * self.sys.n_inputs + iu
                
                Mu[k, i:j] = mu
                Mu_abs[k] = mu_abs
                
            for ipar, ppar in enumerate(par):
                # get index range of `xx` in vector of all indep vars
                i,j = indic[ppar]

                # get dependence vector for the collocation point and spline variable
                mp, mp_abs = self.get_dependence_vectors_p(p, ipar) # actually it is no need to call the function since mp is always 1.0 and mp_abs always 0.

                k = ip * self.sys.n_par + ipar
                
                Mp[k, i:j] = mp # mp = 1
                Mp_abs[k] = mp_abs    # mp_abs = 0

        MC = Container()
        MC.Mx = Mx
        MC.Mx_abs = Mx_abs
        MC.Mdx = Mdx
        MC.Mdx_abs = Mdx_abs
        MC.Mu = Mu
        MC.Mu_abs = Mu_abs
        MC.Mp = Mp
        MC.Mp_abs = Mp_abs
       
        # return Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs, Mp, Mp_abs
        return MC
    def get_dependence_vectors_p(self, p, ipar):
        dep_array_k = np.array([1.0]) # dep_array_k is always 1 for p[0]=k
        dep_array_k_abs = np.array([0.0]) # dep_array_k_abs is always 0 for p[0]=k

        if np.size(p) > 1:
            raise NotImplementedError()

            # determine the spline part to evaluate
    ##!!        i = int(np.floor(t * self.trajectories.n_parts_x / self.trajectories.sys.b))
    ##!!        # h = (self.trajectories.sys.b - self.trajectories.sys.a) / float(self.trajectories.n_parts_x)
    ##!!        if i == self.trajectories.n_parts_x: i -= 1

        tt = np.array([1.0]) ## tt = [1] * par[0]
        dep_vec_k = np.dot(tt, dep_array_k[0])
        dep_vec_abs_k = np.dot(tt, dep_array_k_abs[0])
        return dep_vec_k, dep_vec_abs_k


    # def get_dependence_vectors_p(self, p, ipar, f_inv_psi):
    #     if ipar == 0: # z_par_1
    #         yk = f_inv_psi(self.guess[-self.sys.n_par])
    #         dep_array_k = np.array([self.guess[-self.sys.n_par] / yk])
    #         dep_array_k_abs = np.array([0.0])
    #
    #     else:
    #         dep_array_k = np.array([1.0]) # dep_array_k is always 1 for p[0]=k
    #         dep_array_k_abs = np.array([0.0]) # dep_array_k_abs is always 0 for p[0]=k
    #
    #     if np.size(p) > 1:
    #         raise NotImplementedError()
    #
    #         # determine the spline part to evaluate
    # ##!!        i = int(np.floor(t * self.trajectories.n_parts_x / self.trajectories.sys.b))
    # ##!!        # h = (self.trajectories.sys.b - self.trajectories.sys.a) / float(self.trajectories.n_parts_x)
    # ##!!        if i == self.trajectories.n_parts_x: i -= 1
    #
    #     tt = np.array([1.0]) ## tt = [1] * par[0]
    #     dep_vec_k = np.dot(tt, dep_array_k[0])
    #     dep_vec_abs_k = np.dot(tt, dep_array_k_abs[0])
    #     return dep_vec_k, dep_vec_abs_k

    def get_guess(self, f_inv_psi):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrary values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        '''
        if not self.trajectories._old_splines:
            # user doesn't define initial value of free coefficients
            if self._first_guess is None:
                free_vars_all = np.hstack(self.trajectories.indep_vars.values()) ##:: self.trajectories.indep_vars.values() contains all the free-par. (5*11), free_coeffs_all = array([cx3_0_0, cx3_1_0, ..., cx3_8_0, cx1_0_0, ..., cx1_14_0, cx1_15_0, cx1_16_0, k]
                guess = 0.1 * np.ones(free_vars_all.size) ##:: init. guess = 0.1
                ##!! itemindex = np.argwhere(free_coeffs_all == sp.symbols('k'))
                # Todo: change guess to guess[-n_par:]
                guess[-self.sys.n_par:] = self._parameters['z_par']
                if self.constraints != None:
                    for k, v in self.constraints.items():
                        # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
                        # and the saturation limits y0, y1
                        if type(k) == str:
                            #  park = self.sys.par[int(k[4])]
                            yk_pos = int(k[4])
                            guess[-(self.sys.n_par+yk_pos)] = f_inv_psi(guess[-(self.sys.n_par+yk_pos)]) ##::  yk

                #guess[-1] = self._parameters['z_par'] # in 1st round, the last element of guess is the value of z_par

                from IPython import embed as IPS
                #IPS()
                ##!! self.itemindex = itemindex[0][0]
                ##!! p = np.array([2.5])
                ##!! guess = np.hstack((guess,p[0])
            # user defines initial value of free coefficients
            else:
                guess = np.empty(0)
            
                for k, v in sorted(self.trajectories.indep_vars.items(), key = lambda (k, v): k):
                    logging.debug("Get new guess for spline {}".format(k))

                    if self._first_guess.has_key(k):
                        s = self.trajectories.splines[k]
                        f = self._first_guess[k]

                        free_vars_guess = s.interpolate(f)

                    elif self._first_guess.has_key('seed'):
                        np.random.seed(self._first_guess.get('seed'))
                        free_vars_guess = np.random.random(len(v))
                        
                    else:
                        free_vars_guess = 0.1 * np.ones(len(v))

                    guess = np.hstack((guess, free_vars_guess))
                    guess[-self.sys.n_par:] = self._parameters['z_par']
                    if self.constraints != None:
                        for k, v in self.constraints.items():
                            # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
                            # and the saturation limits y0, y1
                            if type(k) == str:
                                #  park = self.sys.par[int(k[4])]
                                yk_pos = int(k[4])
                                guess[-(self.sys.n_par+yk_pos)] = f_inv_psi(guess[-(self.sys.n_par+yk_pos)]) ##::yk
        else:
            guess = np.empty(0)
            guess_add_finish = False
            # now we compute a new guess for every free coefficient of every new (finer) spline
            # by interpolating the corresponding old (coarser) spline
            for k, v in sorted(self.trajectories.indep_vars.items(), key = lambda (k, v): k):
                if guess_add_finish == False: # must sure that 'self.sys.par' is the last one for 'k'
                    # TODO_ok: introduce a parameter `ku` (factor for increasing spline resolution for u)
                    # formerly its spline resolution was constant
                    # (from that period stems the following if-statement)
                    # currently the input is handled like the states
                    # thus the else branch is switched off
                    if (self.sys.states.__contains__(k) or self.sys.inputs.__contains__(k)):
                        spline_type = self.trajectories.splines[k].type
                    elif (self.sys.par.__contains__(k)):
                        spline_type = 'p'


                    if (spline_type == 'x') or (spline_type == 'u'):
                        logging.debug("Get new guess for spline {}".format(k))

                        s_new = self.trajectories.splines[k]
                        s_old = self.trajectories._old_splines[k]

                        df0 = s_old.df(self.sys.a)
                        dfn = s_old.df(self.sys.b)

                        free_vars_guess = s_new.interpolate(s_old.f, m0=df0, mn=dfn)
                        guess = np.hstack((guess, free_vars_guess))

                    elif (spline_type == 'p' ):#  if self.sys.par is not the last one, then add (and guess_add_finish == False) here.
                        guess = np.hstack((guess, self.sol[-self.sys.n_par:])) # sequence of guess is (u,x,p) ##:: yk
                        #guess = np.hstack((guess, [1, 2]))
                        guess_add_finish = True
        # the new guess
        self.guess = guess
    
    
    def solve(self, G, DG):
        '''
        This method is used to solve the collocation equation system.
        
        Parameters
        ----------
        
        G : callable
            Function that "evaluates" the equation system.
        
        DG : callable
            Function for the jacobian.
        '''
        logging.debug("Solving Equation System")
        
        # create our solver
        self.solver = Solver(F=G, DF=DG, x0=self.guess, ##:: x0 = [u,x,z_par]
                             tol=self._parameters['tol'],
                             reltol=self._parameters['reltol'],
                             maxIt=self._parameters['sol_steps'],
                             method=self._parameters['method'],
                             par=np.array(self.guess[-self.sys.n_par:])) ##!! , itemindex = self.itemindex # par_k

        # solve the equation system
        self.sol, par, k_list = self.solver.solve()
        return self.sol, par, k_list

    def save(self):

        save = dict()

        # parameters
        save['parameters'] = self._parameters

        # vector field and jacobian
        save['f'] = self._f
        save['Df'] = self._Df

        # guess
        save['guess'] = self.guess
        
        # solution
        save['sol'] = self.sol
    
        # k
        save['z_par'] = self.sol[-self.sys.n_par]

        return save

def collocation_nodes(a, b, npts, coll_type):
    '''
    Create collocation points/nodes for the equation system.
    
    Parameters
    ----------
    
    a : float
        The left border of the considered interval.
    
    b : float
        The right border of the considered interval.
    
    npts : int
        The number of nodes.
    
    coll_type : str
        Specifies how to generate the nodes.
    
    Returns
    -------
    
    numpy.ndarray
        The collocation nodes.
    
    '''
    if coll_type == 'equidistant':
        # get equidistant collocation points
        cpts = np.linspace(a, b, npts, endpoint=True)
    elif coll_type == 'chebychev':
        # determine rank of chebychev polynomial
        # of which to calculate zero points
        nc = int(npts) - 2

        # calculate zero points of chebychev polynomial --> in [-1,1]
        cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
        cheb_cpts.sort()

        # transfer chebychev nodes from [-1,1] to our interval [a,b]
        chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

        # add left and right borders
        cpts = np.hstack((a, chpts, b))
    else:
        logging.warning('Unknown type of collocation points.')
        logging.warning('--> will use equidistant points!')
        cpts = np.linspace(a, b, npts, endpoint=True)
    
    return cpts

def _get_derivation_order(fnc):
    '''
    Returns derivation order of function according to place in integrator chain.
    '''

    from .splines import Spline
    
    if fnc.im_func == Spline.f.im_func:
        return 0
    elif fnc.im_func == Spline.df.im_func:
        return 1
    elif fnc.im_func == Spline.ddf.im_func:
        return 2
    elif fnc.im_func == Spline.dddf.im_func:
        return 3
    else:
        raise ValueError()

def _build_sol_from_free_coeffs(splines):
    '''
    Concatenates the values of the independent coeffs
    of all splines in given dict to build pseudo solution.
    '''

    sol = np.empty(0)
    for k, v in sorted(splines.items(), key = lambda (k, v): k):
        assert not v._prov_flag
        sol = np.hstack([sol, v._indep_coeffs])

    return sol

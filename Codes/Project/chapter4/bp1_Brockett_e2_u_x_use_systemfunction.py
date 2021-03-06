# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 15:18:44 2017

u as Polynomform and uses solve.integrator to get x (for Brockett example)

@author: rst
"""

import sympy as sp
import numpy as np
sp.init_printing()
from scipy.integrate import ode
from IPython import embed as IPS

class Y_ode(object):#
    def __init__(self):
        self.dt = 0.01
        a0,a1,a2,b0,b1,b2 = sp.symbols('a_0 a_1 a_2 b_0 b_1 b_2')
        alpha,beta,gamma = sp.symbols('alpha beta gamma')


        a0,a1,b0 = 1,1,1 # freie Variable
        alpha,beta,gamma = 1,1,1

        eq1 = (a0+0.5*a1+1.0/3*a2+alpha)
        eq2 = (b0+0.5*b1+1.0/3*b2+beta)
        eq3 = (beta*a0-alpha*b0) + 1.0/2*(beta*a1-alpha*b1) + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2)+gamma
        eq_list = [eq1,eq2,eq3]
        x_list = [a2,b1,b2]
        S = sp.solve(eq_list, x_list)
        self.X1_List = []
        self.X2_List = []
        self.X3_List = []
        self.U1_List = []
        self.U2_List = []
        self.t_List =  []

        a2 = S[0][0]
        b1 = S[0][1]
        b2 = S[0][2]

        self.solver = ode(self.rhs)
        self.solver.set_initial_value([alpha,beta,gamma])
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)

        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

    def u(self,t):
        u1 = self.a0 + self.a1*t + self.a2*t**2
        u2 = self.b0 + self.b1*t + self.b2*t**2
        u = [u1,u2]
        #IPS()
        return u
        # self.u = u
        
    def f(self,x,u):
        x1,x2,x3 = x
        u1,u2 = u
        ff = [u1,u2,x2*u1-x1*u2]
        return ff
        

    def rhs(self,t,x):
        u = self.u(t)
        # print ('u={},t={}'.format(u,t))
        dx = self.f(x,u)
        return dx

    def sim(self):
        t=0
        while t<1:
            #IPS()
            x = list(self.solver.integrate(self.solver.t+self.dt))
            # print ('t = {}, x={}'.format(self.solver.t, x))
            t = round(self.solver.t, 5)
            self.t_List.append(t)
            self.X1_List.append(x[0])
            self.X2_List.append(x[1])
            self.X3_List.append(x[2])
            self.U1_List.append(self.u(t)[0])
            self.U2_List.append(self.u(t)[1])

            X_List = [self.X1_List, self.X2_List, self.X3_List]
            U_List = [self.U1_List, self.U2_List]
        return X_List, U_List, self.t_List


y_ode = Y_ode()
sol_xt,sol_ut,sol_t = y_ode.sim()
print('xt={}'.format(sol_xt))
print('ut={}'.format(sol_ut))
print(sol_t)

# from IPython import embed as IPS
# IPS()
plot = True # using Refsol
if plot:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import rc

    matplotlib.rc('xtick', labelsize=50)#44
    matplotlib.rc('ytick', labelsize=50)
    matplotlib.rcParams.update({'font.size': 50})
    matplotlib.rcParams['xtick.major.pad'] = 12
    matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    t = sol_t
    x1 = sol_xt[0]
    x2 = sol_xt[1]
    x3 = sol_xt[2]
    u1 = sol_ut[0]

    plt.figure(1)
    plt.grid()
    nx = 2
    if len(sol_xt) % 2 == 0:  # a.size
        mx = len(sol_xt) / nx
    else:
        mx = len(sol_xt) / nx + 1

    ax = xrange(len(sol_xt))

    for i in ax:
        plt.subplot(mx, nx, i + 1)
        plt.plot(t, sol_xt[i], linewidth=3.0)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.grid()
        if i == 0:
            plt.ylabel(r'$x_{}$'.format(i + 1), fontsize=46)
        if i == 1:
            plt.ylabel(r'$x_{}$'.format(i + 1), fontsize=46)
        if i == 2:
            plt.ylabel(r'$x_{}$'.format(i + 1), fontsize=46)
        if i == 3:
            plt.ylabel(r'$x_{}$'.format(i + 1), fontsize=46)

    plt.figure(2)

    nx = 2
    if len(sol_ut) % 2 == 0:  # a.size
        mx = len(sol_ut) / nx
    else:
        mx = len(sol_ut) / nx + 1

    ax = xrange(len(sol_ut))

    for i in ax:
        plt.subplot(mx, nx, i + 1)
        plt.plot(t, sol_ut[i], linewidth=3.0)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=52)#46
        plt.grid()
        if i == 0:
            plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=52)
        if i == 1:
            plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=52)
        if i == 2:
            plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=52)
        if i == 3:
            plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=52)


    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    #
    # plt.sca(ax1)
    # plt.plot(t, u1, linewidth=3.0)
    # plt.xlabel(r'$t(s)$', fontsize=46)
    # plt.ylabel(r'$u_{1}$', fontsize=46)
    # plt.grid()
    # plt.sca(ax2)
    # plt.plot(range(len(S.k_list)), S.k_list, '.',markersize = 10, linewidth=3.0)
    # plt.xlabel(r'$Iteration-Mal$', fontsize=46)
    # plt.ylabel('$k$', fontsize=46)
    # plt.grid()
    plt.show()













import xlwt # save u and x in Excel
if 0:
    workbook = xlwt.Workbook()  # encoding='utf-8'
    booksheetU1 = workbook.add_sheet('U1', cell_overwrite_ok=False)
    for i, row in enumerate(sol_ut[0]):
        for j, col in enumerate(row):
            booksheetU1.write(i, j, float(col))

    booksheetU2 = workbook.add_sheet('U2', cell_overwrite_ok=False)
    for i, row in enumerate(sol_ut[1]):
        for j, col in enumerate(row):
            booksheetU2.write(i, j, float(col))

    booksheetX1 = workbook.add_sheet('X1', cell_overwrite_ok=False )
    for i, row in enumerate(sol_xt[0]):
        for j, col in enumerate(row):
            booksheetX1.write(i, j, float(col))

    booksheetX2 = workbook.add_sheet('X2', cell_overwrite_ok=False)
    for i, row in enumerate(sol_xt[1]):
        for j, col in enumerate(row):
            booksheetX2.write(i, j, float(col))

    booksheetX3 = workbook.add_sheet('X3', cell_overwrite_ok=False)
    for i, row in enumerate(sol_xt[2]):
        for j, col in enumerate(row):
            booksheetX3.write(i, j, float(col))

    # booksheetT_time = workbook.add_sheet('T_time', cell_overwrite_ok=False)
    # for i, row in enumerate(T_time):
    #     booksheetT_time.write(i, 0, row)

    workbook.save('d:\\Brockett_e2_u_Asymptotically_stable_U_List_use_systemfunction.xls')

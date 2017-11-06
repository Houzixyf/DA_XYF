# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 17:31:55 2017
使用 solve.integrator来得到x,并且判断渐近稳定性。u是三阶多项式。
@author: rst
"""

import sympy as sp
import numpy as np
import time
sp.init_printing()
from scipy.integrate import ode
from IPython import embed as IPS

class Y_ode(object):#
    def __init__(self, alpha, beta, gamma, a0, a1, b0):
        self.dt = 0.01
        # a0,a1,a2,b0,b1,b2 = sp.symbols('a_0 a_1 a_2 b_0 b_1 b_2')
        # alpha,beta,gamma = sp.symbols('alpha beta gamma')
        a2, b1, b2 = sp.symbols('a_2, b_1, b_2')

        a0,a1,b0 = a0, a1, b0 # 自变量
        alpha,beta,gamma = alpha, beta, gamma

        eq1 = (a0+0.5*a1+1.0/3*a2+alpha)
        eq2 = (b0+0.5*b1+1.0/3*b2+beta)
        eq3 = (beta*a0-alpha*b0) + 1.0/2*(beta*a1-alpha*b1) + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2)+gamma
        eq_list = [eq1,eq2,eq3]
        x_list = [a2,b1,b2]#因变量
        S = sp.solve(eq_list, x_list)


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
        xt_list = []
        ut_list = []
        t_list = []
        while t<1:
            #IPS()
            x = list(self.solver.integrate(self.solver.t+self.dt))
            # print ('t = {}, x={}'.format(self.solver.t, x))
            t = round(self.solver.t, 5)
            u = self.u(t)
            xt_list.append(x)
            ut_list.append(u)
            t_list.append(t)
        return xt_list, ut_list, t_list






X1_List = []
X2_List = []
X3_List = []
U1_List = []
U2_List = []
T_time = []

alpha, beta, gamma = 0, 0, 0
a0, a1, b0 = 1, 1, 1  # 自变量

delta_alpha = 0.1
delta_beta = 0.1
delta_gamma = 0.1

# delta_alpha = np.array([0.1, 0, 0])
# delta_beta = np.array([0, 0.1, 0])
# delta_gamma = np.array([0, 0, 0.1])



for d1 in [-1, 0, 1]:
    for d2 in [-1, 0, 1]:
        for d3 in [-1, 0, 1]:
            T_start = time.time()
            a2, b1, b2 = sp.symbols('a_2 b_1 b_2')
            alpha_new = alpha + d1 * delta_alpha
            beta_new = beta + d2 * delta_beta
            gamma_new = gamma + d3 * delta_gamma
            y_ode = Y_ode(alpha=alpha_new, beta=beta_new,gamma=gamma_new,a0=a0,a1=a1,b0=b0)
            xt,ut,t = y_ode.sim()
            T_end = time.time()
            T_time.append(T_end - T_start)


            # print('xt={}'.format(xt))
            # print('ut={}'.format(ut))

            xt = np.array(xt)
            ut = np.array(ut)
            X1_List.append(xt[:,0])
            X2_List.append(xt[:,1])
            X3_List.append(xt[:,2])
            U1_List.append(ut[:,0])
            U2_List.append(ut[:,1])

            X_List = [X1_List, X2_List, X3_List]
            U_List = [U1_List, U2_List]



plot = 1
if plot:
    import matplotlib.pyplot as plt
    t = t
    plt.figure(1)
    nx = 2
    xa = [0,0,0]
    if len(xa) % 2 == 0: # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))
    # from IPython import embed as IPS
    # IPS()
    for i in ax:
        for j in range(len(X_List[0])):
            plt.subplot(mx,nx,i+1)
            plt.plot(t,X_List[i][j])
            # plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$x_{}$'.format(i+1))

    plt.figure(2)
    ua = [1,1]
    if len(ua) % 2 == 0:
        nu = 2
        mu = len(ua) / nu
    elif len(ua) == 1:
        nu = 1
        mu = 1
    else:
        nu = 2
        mu = len(ua) / nu + 1

    ax = xrange(len(ua))

    for i in ax:
        for j in range(len(U_List[0])):
            plt.subplot(mu, nu, i + 1)
            plt.plot(t, U_List[i][j])
    #     plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$u_{}$'.format(i + 1))

    plt.show()







import xlwt # save u in Excel
if 1:
    workbook = xlwt.Workbook()  # encoding='utf-8'
    booksheetU1 = workbook.add_sheet('U1', cell_overwrite_ok=False)
    for i, row in enumerate(U1_List):
        for j, col in enumerate(row):
            booksheetU1.write(i, j, float(col))

    booksheetU2 = workbook.add_sheet('U2', cell_overwrite_ok=False)
    for i, row in enumerate(U2_List):
        for j, col in enumerate(row):
            booksheetU2.write(i, j, float(col))

    booksheetX1 = workbook.add_sheet('X1', cell_overwrite_ok=False )
    for i, row in enumerate(X1_List):
        for j, col in enumerate(row):
            booksheetX1.write(i, j, float(col))

    booksheetX2 = workbook.add_sheet('X2', cell_overwrite_ok=False)
    for i, row in enumerate(X2_List):
        for j, col in enumerate(row):
            booksheetX2.write(i, j, float(col))

    booksheetX3 = workbook.add_sheet('X3', cell_overwrite_ok=False)
    for i, row in enumerate(X3_List):
        for j, col in enumerate(row):
            booksheetX3.write(i, j, float(col))

    booksheetT_time = workbook.add_sheet('T_time', cell_overwrite_ok=False)
    for i, row in enumerate(T_time):
        booksheetT_time.write(i, 0, row)

    workbook.save('d:\\Brockett_e2_u_Asymptotically_stable_U_List_use_systemfunction.xls')

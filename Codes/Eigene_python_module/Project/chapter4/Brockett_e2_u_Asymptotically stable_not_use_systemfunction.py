# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 12:15:54 2017
解Brockett例2，x的初始值为原点，终值为(alpha,beta,gamma),三个方程6个变量。T = 1s。 设计u(t),验证渐近稳定性。
@author: rst
"""

import sympy as sp
import numpy as np
import time
sp.init_printing()
a0,a1,a2,b0,b1,b2 = sp.symbols('a_0 a_1 a_2 b_0 b_1 b_2')
alpha,beta,gamma = sp.symbols('alpha beta gamma')
alpha,beta,gamma = 1,1,1
a0,a1,b0 = 1,1,1 # 自变量

delta_alpha = 0.1
delta_beta = 0.1
delta_gamma = 0.1

#delta_alpha = np.array([0.1, 0, 0])
#delta_beta = np.array([0, 0.1, 0])
#delta_gamma = np.array([0, 0, 0.1])

T_time = []
U1_List = []
U2_List = []
X1_List = []
X2_List = []
X3_List = []
D = []


for d1 in [-1, 0, 1]:
    for d2 in [-1, 0, 1]:
        for d3 in [-1, 0, 1]:
            
            T_start = time.time()
            a2, b1, b2 = sp.symbols('a_2 b_1 b_2')
            alpha_new = alpha + d1 * delta_alpha
            beta_new = beta + d2 * delta_beta
            gamma_new = gamma + d3 * delta_gamma
            
            eq1 = (a0+0.5*a1+1.0/3*a2+alpha_new)
            eq2 = (b0+0.5*b1+1.0/3*b2+beta_new)
            eq3 = (beta_new*a0-alpha_new*b0) + 1.0/2*(beta_new*a1-alpha_new*b1) + 1.0/3*(1.0/2*a1*b0+beta_new*a2-1.0/2*a0*b1-alpha_new*b2)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2)+gamma_new
            eq_list = [eq1,eq2,eq3]
            x_list = [a2,b1,b2]#因变量
            S = sp.solve(eq_list, x_list)
            
            # print(sp.latex(S))
            t = np.linspace(0,1,20)
            a2 = S[0][0]
            b1 = S[0][1]
            b2 = S[0][2]
            
            u1 = a0 + a1*t +a2*t**2

            u2 = b0 + b1*t +b2*t**2
            
            x1 = a0*t + 1.0/2*a1*t**2 + 1.0/3*a2*t**3 + alpha_new

            x2 = b0*t + 1.0/2*b1*t**2 + 1.0/3*b2*t**3 + beta_new

            x3 = (beta_new*a0-alpha_new*b0)*t + 1.0/2*(beta_new*a1-alpha_new*b1)*t**2 + 1.0/3*(1.0/2*a1*b0+beta_new*a2-1.0/2*a0*b1-alpha_new*b2)*t**3+1.0/6*(a2*b0-a0*b2)*t**4+1.0/30*(a2*b1-a1*b2)*t**5+gamma_new

            T_end = time.time()
            T_time.append(T_end - T_start)
            U1_List.append(u1)
            U2_List.append(u2)
            X1_List.append(x1)
            X2_List.append(x2)
            X3_List.append(x3)
            D.append([d1, d2, d3])






# plot
#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.plot(t,u1)
## plt.title()
#plt.xlabel('t')
#plt.ylabel(r'$u_{1}$')
#
#plt.figure(2)
#plt.plot(t,u2)
## plt.title()
#plt.xlabel('t')
#plt.ylabel(r'$u_{2}$')
#
#plt.figure(3)
#plt.subplot(2,2,1)
#plt.plot(t,x1)
#plt.xlabel('t')
#plt.ylabel(r'$x_{1}$')
#
#plt.subplot(2,2,2)
#plt.plot(t,x2)
#plt.xlabel('t')
#plt.ylabel(r'$x_{2}$')
#
#plt.subplot(2,2,3)
#plt.plot(t,x3)
#plt.xlabel('t')
#plt.ylabel(r'$x_{3}$')
#plt.show()
#
#print u1
#print u2



import xlwt # save u in Excel

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

workbook.save('Brockett_e2_u_Asymptotically_stable_U_List_not_use_systemfunction.xls')














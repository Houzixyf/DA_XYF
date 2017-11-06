# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 14:13:27 2017
解Brockett例2，x的初始值为原点，终值为(alpha,beta,gamma),三个方程6个变量。设计u(t)。
@author: rst
"""

import sympy as sp
sp.init_printing()
a0,a1,a2,b0,b1,b2 = sp.symbols('a_0 a_1 a_2 b_0 b_1 b_2')
alpha,beta,gamma = sp.symbols('alpha beta gamma')


#a0,a1,b0 = 1,1,1 # 自变量
#alpha,beta,gamma = 0.9,1.1,0.9

#eq1 = sp.Eq(alpha, a0+0.5*a1+1.0/3*a2)
#eq2 = sp.Eq(beta, b0+0.5*b1+1.0/3*b2)
#eq3 = sp.Eq(gamma, 1.0/6*(a1*b0-a0*b1)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2))

eq1 = (a0+0.5*a1+1.0/3*a2+alpha)
eq2 = (b0+0.5*b1+1.0/3*b2+beta)
eq3 = (beta*a0-alpha*b0) + 1.0/2*(beta*a1-alpha*b1) + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2)+gamma
eq_list = [eq1,eq2,eq3]
x_list = [a2,b1,b2]#因变量
S = sp.solve(eq_list, x_list)
S_num = sp.solve(eq_list,x_list)
#S = solve([3 * y + 5 * y - 19, 4 * x - 3 * y - 6],[x,y]))
print(sp.latex(S))

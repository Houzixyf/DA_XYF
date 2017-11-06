# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 14:29:39 2017
解Brockett例2，x的初始值为原点，终值为(alpha,beta,gamma),三个方程6个变量。T = 1s。 设计u(t),用u(t)来推出x(t)。
@author: rst
"""

import sympy as sp
import numpy as np
sp.init_printing()
a0,a1,a2,b0,b1,b2 = sp.symbols('a_0 a_1 a_2 b_0 b_1 b_2')
alpha,beta,gamma = sp.symbols('alpha beta gamma')


a0,a1,b0 = 1,1,1 # 自变量

alpha,beta,gamma = 1,1,1

eq1 = (a0+0.5*a1+1.0/3*a2+alpha)
eq2 = (b0+0.5*b1+1.0/3*b2+beta)
eq3 = (beta*a0-alpha*b0) + 1.0/2*(beta*a1-alpha*b1) + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)+1.0/6*(a2*b0-a0*b2)+1.0/30*(a2*b1-a1*b2)+gamma
eq_list = [eq1,eq2,eq3]
x_list = [a2,b1,b2]#因变量
S = sp.solve(eq_list, x_list)
S_num = sp.solve(eq_list,x_list)

# print(sp.latex(S))
t = np.linspace(0,1,20)
a2 = S[0][0]
b1 = S[0][1]
b2 = S[0][2]

u1 = a0 + a1*t +a2*t**2



u2 = b0 + b1*t +b2*t**2




x1 = a0*t + 1.0/2*a1*t**2 + 1.0/3*a2*t**3 + alpha



x2 = b0*t + 1.0/2*b1*t**2 + 1.0/3*b2*t**3 + beta



x3 = (beta*a0-alpha*b0)*t + 1.0/2*(beta*a1-alpha*b1)*t**2 + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)*t**3+1.0/6*(a2*b0-a0*b2)*t**4+1.0/30*(a2*b1-a1*b2)*t**5+gamma





def u1_test(t):
    u1 = a0 + a1*t +a2*t**2
    return u1

def u2_test(t):
    u2 = b0 + b1*t +b2*t**2
    return u2

def x1_test(t):
    x1 = a0*t + 1.0/2*a1*t**2 + 1.0/3*a2*t**3 + alpha
    return x1

def x2_test(t):
    x2 = b0*t + 1.0/2*b1*t**2 + 1.0/3*b2*t**3 + beta
    return x2

def x3_test(t):
    x3 = (beta*a0-alpha*b0)*t + 1.0/2*(beta*a1-alpha*b1)*t**2 + 1.0/3*(1.0/2*a1*b0+beta*a2-1.0/2*a0*b1-alpha*b2)*t**3+1.0/6*(a2*b0-a0*b2)*t**4+1.0/30*(a2*b1-a1*b2)*t**5+gamma
    return x3


# plot
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(t,u1)
# plt.title()
plt.xlabel('t')
plt.ylabel(r'$u_{1}$')

plt.figure(2)
plt.plot(t,u2)
# plt.title()
plt.xlabel('t')
plt.ylabel(r'$u_{2}$')

plt.figure(3)
plt.subplot(2,2,1)
plt.plot(t,x1)
plt.xlabel('t')
plt.ylabel(r'$x_{1}$')

plt.subplot(2,2,2)
plt.plot(t,x2)
plt.xlabel('t')
plt.ylabel(r'$x_{2}$')

plt.subplot(2,2,3)
plt.plot(t,x3)
plt.xlabel('t')
plt.ylabel(r'$x_{3}$')
plt.show()

print u1
print u2




















# -*- coding: utf-8 -*-
"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log, auxiliary
import numpy as np
from sympy import cos, sin
from pytrajectory import penalty_expression as pe
import time
import pickle

log.console_handler.setLevel(10)
# first, we define the function that returns the vectorfield


# 加上pe()的好处是最终的结果中，输入u的值会比较小，说明用一个较小的输入就可以控制系统； 但是 k 的范围依然需要手动确定，并不是给任意一个 k 都可以得到结果。



def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2, x3, x4 = x  # state variables
    u1, = u  # input variable

    e = 0.9  # inertia coupling

    s = sin(x3)
    c = cos(x3)

    ff = [         x2,
                   u1,
                   x4,
                   -e * x2 ** 2 * s - (1 + e * c) * u1
                   ]

    ff = [k * eq for eq in ff]

    if evalconstr:
            res = 1*pe(k, 0.1, 15) #  pe(k, 0, 10)
            ff.append(res)

    return ff


xa = [  0.0,
        0.0,
        0.4*np.pi,
        0.0]

xb = [  0.2*np.pi,
        0.0,
        0.2*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

a = 0.0
b = 1.0
par = [1.5]


T_time=[]
SP = []
Time_SP = []
Reached_Accuracy = []
U_Umgebung = []
Delta_U = []
D = []

# use_refsol = False # use refsol
for i in range(1):
    first_guess = {'seed': 5}


    # now we create our Trajectory object and alter some method parameters via the keyword arguments
    # S = ControlSystem(f, a, b, xa, xb, ua, ub,su=20, sx=10, kx=3, use_chains=False, k=par, sol_steps=100, dt_sim = 0.01, first_guess=first_guess, maxIt=2)  # k must be a list,  k=par, refsol=refsol, first_guess=first_guess,

    T_start = time.time()

    delta_xx_1 = np.array([np.pi / 180*100, 0, 0, 0])
    delta_xx_2 = np.array([0, 0.01*100, 0, 0])
    delta_xx_3 = np.array([0, 0, np.pi / 180*100, 0])
    delta_xx_4 = np.array([0, 0, 0, 0.01*100])
    for d1 in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            for d3 in [-1, 0, 1]:
                for d4 in [-1, 0, 1]:
                    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
                        continue
                    x_change = d1 * delta_xx_1 + d2 * delta_xx_2 + d3 * delta_xx_3 + d4 * delta_xx_4

                    print('*********************')
                    print('Begint the new loop')
                    print('*********************')

                    par = [1.5] # par must re-init
                    dt_sim = 0.001
                    S = ControlSystem(f, a, b, xa, xb, ua, ub, su=20, sx=10, kx=3, use_chains=False, k=par, sol_steps=100,  first_guess=first_guess, maxIt=2, dt_sim=dt_sim) # dt_sim=0.01,
                    x, u, par = S.solve(x_change)
                    T_end = time.time()
                    print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))

                    if 1: # plot for u at last
                        import matplotlib.pyplot as plt
                        t_num = (1.0-0.99)/dt_sim +1
                        t_last_list = S.sim_data[0][-t_num:]
                        u = S.sim_data[2][:, 0][-t_num:]
                        x = S.sim_data[1][:, 0][-t_num:]
                        plt.figure()
                        plt.plot(t_last_list, x)
                        plt.xlabel('t')
                        plt.ylabel(r'$u_{}$'.format(1))
                        plt.show()


                    t_num = (1.0 - 0.9) / dt_sim + 1
                    u_last = S.sim_data[2][:, 0][-t_num:]
                    T_time.append(T_end-T_start)
                    SP.append(S.nIt)
                    Reached_Accuracy.append(S.reached_accuracy)
                    Time_SP.append([T_time[i],SP[i],Reached_Accuracy[i]])
                    U_Umgebung.append(u_last)
                    Delta_U.append([[d1,d2,d3,d4], x_change, u_last])
                    D.append([d1,d2,d3,d4])
                    print ('Time to solve with seed-{}(s): {}'.format(i, T_time[-1]))
# print ('delta,x_change and u:{}').format(Delta_U)
import xlwt
workbook=xlwt.Workbook()# encoding='utf-8'
booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=False)
for i,row in enumerate(U_Umgebung):
    for j,col in enumerate(row):
        booksheet.write(i,j,col)
workbook.save('d:\\U_Umgebung.xls')

booksheet2=workbook.add_sheet('Sheet 2', cell_overwrite_ok=True)

for i,row in enumerate(D):
    for j,col in enumerate(row):
        booksheet.write(i,j,col)
workbook.save('d:\\D.xls')











plot = False
if plot:# plot for x and u
    import matplotlib.pyplot as plt
    t = S.sim_data[0]
    print t
    plt.figure(1)
    nx = 2
    if len(xa) % 2 == 0: # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for i in ax:
        plt.subplot(mx,nx,i+1)
        plt.plot(t,S.sim_data[1][:, i])
        # plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$x_{}$'.format(i+1))

    plt.figure(2)
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
        plt.subplot(mu, nu, i + 1)
        #plt.xlim(0,0.051)
        plt.plot(t, S.sim_data[2][:, i])
    #     plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$u_{}$'.format(i + 1))

    plt.show()
print '\n'
print ('Time, Number of Iteration, Reached accuracy or not: {}').format(Time_SP)
from IPython import embed as IPS
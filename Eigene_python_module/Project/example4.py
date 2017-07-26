# -*- coding: utf-8 -*-
"""
# underactuated manipulator


"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
import numpy as np
from sympy import cos, sin
from pytrajectory import penalty_expression as pe
import time
log.console_handler.setLevel(10)
from IPython import embed as IPS
from pytrajectory.auxiliary import Container
import pickle
# first, we define the function that returns the vectorfield


# 加上pe()的好处是最终的结果中，输入u的值会比较小，说明用一个较小的输入就可以控制系统； 但是 k 的范围依然需要手动确定，并不是给任意一个 k 都可以得到结果。

def f(x, u, par, evalconstr=True):
    k, = par
    x1, x2, x3, x4 = x  # state variables
    u1, = u  # input variable

    e = 0.9  # inertia coupling

    s = sin(x3)
    c = cos(x3)

    ff = [x2,
          u1,
          x4,
          -e * x2 ** 2 * s - (1 + e * c) * u1
          ]

    ff = [k * eq for eq in ff]

    if evalconstr:
        res = 1 * pe(k, 0.1, 15)  # pe(k, 0.1, 15) -> k=11s
        ff.append(res)

    return ff


xa = [0.0,
      0.0,
      0.4 * np.pi,
      0.0]

xb = [0.2 * np.pi,
      0.0,
      0.2 * np.pi,
      0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

a = 0.0
b = 1.0
par = [1.5]

time = 0.99

path = 'E:\Yifan_Xue\DA\Data\Data_for_Brockett_pe(k,0.1,15)_t_0.99'
use_refsol = True
if use_refsol:
    refsol_x_place = open(path + '\\x_refsol.plk', 'rb')
    refsol_x = pickle.load(refsol_x_place)
    refsol_x_place.close()

    refsol_u_place = open(path + '\\u_refsol.plk', 'rb')
    refsol_u = pickle.load(refsol_u_place)
    refsol_u_place.close()

    refsol_t_place = open(path + '\\t_refsol.plk', 'rb')
    refsol_t = pickle.load(refsol_t_place)
    refsol_t_place.close()

    b = round(1.0-time,5)
    xa = refsol_x[0]
    xb = refsol_x[-1]
    ua = refsol_u[0]
    ub = refsol_u[-1]

    Refsol = Container()
    Refsol.tt = refsol_t
    Refsol.xx = refsol_x
    Refsol.uu = refsol_u
    Refsol.n_raise_spline_parts = 0



# T_time = []
# SP = []
# Time_SP = []
# Reached_Accuracy = []


first_guess = {'seed': 5}
dt_sim = 0.001
# now we create our Trajectory object and alter some method parameters via the keyword arguments   refsol=refsol,
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=20, sx=10, kx=3, use_chains=False, k=par, sol_steps=100, dt_sim=dt_sim, refsol=Refsol, maxIt=2, first_guess=None)  # k must be a list,  k=par, refsol=refsol, first_guess=first_guess, first_guess=first_guess,
# T_start = time.time()
x, u, par = S.solve()

save_res = False
if save_res:
    i, = np.where(S.sim_data_tt == time)
    res_x_data = S.sim_data_xx[i[0]:]
    res_x_place = open(path + '\\x_refsol.plk', 'wb')
    pickle.dump(res_x_data, res_x_place)
    res_x_place.close()

    res_u_data = S.sim_data_uu[i[0]:]
    res_u_place = open(path + '\\u_refsol.plk', 'wb')
    pickle.dump(res_u_data, res_u_place)
    res_u_place.close()

    res_t_data = S.sim_data_tt[i[0]:] - time
    res_t_data[-1] = round(res_t_data[-1], 5)
    res_t_place = open(path + '\\t_refsol.plk', 'wb')
    pickle.dump(res_t_data, res_t_place)
    res_t_place.close()



#T_end = time.time()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))

# IPS()
# T_time.append(T_end - T_start)
# SP.append(S.nIt)
# Reached_Accuracy.append(S.reached_accuracy)
# Time_SP.append([T_time[i], SP[i], Reached_Accuracy[i]])
# print ('Time to solve with seed-{}(s): {}'.format(i, T_time[-1]))

plot = True # with Refsol
if plot:
    import matplotlib.pyplot as plt

    t = S.sim_data[0]
        # x1 = S.sim_data[1][i[0]:, 0]
        # x2 = S.sim_data[1][i[0]:, 1]
        # x3 = S.sim_data[1][i[0]:, 2]
        # x4 = S.sim_data[1][i[0]:, 3]
        #
        # u1 = S.sim_data[2][i[0]:, 0]

    plt.figure(1)
    nx = 2
    if len(xa) % 2 == 0:  # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for i in ax:
        plt.subplot(mx, nx, i + 1)
        plt.plot(t, S.sim_data[1][:, i])
            # plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$x_{}$'.format(i + 1))

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
        # plt.xlim(0,0.051)
        plt.plot(t, S.sim_data[2][:, i])
        #     plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$u_{}$'.format(i + 1))
    plt.show()


















plot1 = False # without Refsol
if plot1:
    import matplotlib.pyplot as plt
    i, = np.where(S.sim_data_tt == time)
    t = S.sim_data[0][i[0]:]
        # x1 = S.sim_data[1][i[0]:, 0]
        # x2 = S.sim_data[1][i[0]:, 1]
        # x3 = S.sim_data[1][i[0]:, 2]
        # x4 = S.sim_data[1][i[0]:, 3]
        #
        # u1 = S.sim_data[2][i[0]:, 0]

    plt.figure(1)
    nx = 2
    if len(xa) % 2 == 0:  # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for l in ax:
        plt.subplot(mx, nx, l + 1)
        plt.plot(t, S.sim_data[1][i[0]:, l])
            # plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$x_{}$'.format(l + 1))
        plt.xlim(time, 1.0)

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

    for l in ax:
        plt.subplot(mu, nu, l + 1)
        # plt.xlim(0,0.051)
        plt.plot(t, S.sim_data[2][i[0]:, l])
        #     plt.title()
        plt.xlabel('t')
        plt.ylabel(r'$u_{}$'.format(l + 1))
        plt.xlim(time, 1.0)
    plt.show()
# print '\n'
# print ('Time, Number of Iteration, Reached accuracy or not: {}').format(Time_SP)


from IPython import embed as IPS
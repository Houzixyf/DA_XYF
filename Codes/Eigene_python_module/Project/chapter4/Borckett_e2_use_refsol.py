from pytrajectory import ControlSystem, log
import numpy as np
import time
from sympy import cos, sin
from pytrajectory import penalty_expression as pe
import pickle


def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2, x3 = x       # system state variables
    u1, u2 = u                  # input variable


    ff = [  u1,
            u2,
            x2*u1-x1*u2,
        ]

    ff = [k * eq for eq in ff]

    if evalconstr:
        res = 1*pe(k, 0.1, 5) #  pe(k, 0, 10)
        ff.append(res)
    return ff

a = 0.0
xa_start = [0, 0, 0]

b = 1.0
xb = [0, 0, 0.0]

ua = [0.0, 0.0]
ub = [0.0, 0.0]
par = [1.23]

S_time = 0.9
dt_sim = round((1.0 - S_time) / 10, 5)
dt_sim=dt_sim

path = 'E:\Yifan_Xue\DA\Data\with_Refsol_Brockett\Data_for_Brockett_e2_t_' + str(S_time)

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

    b = round(1.0-S_time,5)
    xa = refsol_x[0]
    xb = refsol_x[-1]
    ua = refsol_u[0]
    ub = refsol_u[-1]

    Refsol = Container()
    Refsol.tt = refsol_t
    Refsol.xx = refsol_x
    Refsol.uu = refsol_u
    Refsol.n_raise_spline_parts = 0





S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100,maxIt= , refsol=Refsol)
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], par))

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
"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem ,log
import numpy as np
from sympy import cos, sin
from pytrajectory import penalty_expression as pe
import pickle

log.console_handler.setLevel(10)


# first, we define the function that returns the vectorfield
def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5  # length of the pendulum rod
    g = 9.81  # gravitational acceleration
    M = 1.0  # mass of the cart
    m = 0.1  # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = [x2,
          u1,
          x4,
          -(1 / l) * (g * sin(x3) + u1 * cos(x3))  # -g/l*s - 1/l*c*u1
          ]

    ff = [k * eq for eq in ff]

    if evalconstr:
        res = pe(k, 0.1, 5)
        ff.append(res)

    return ff


# if 0:
#     from matplotlib import pyplot as plt
#     from ipHelp import IPS
#     import sympy as sp
#     kk = np.linspace(-20, 20)
#     x = sp.Symbol('x')
#     pefnc = sp.lambdify(x, pe(x, 0, 10), modules='numpy')
#     #IPS()
#     plt.plot(kk, pefnc(kk))
#     plt.plot(kk, (kk-5)**2)
#     plt.show()


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, 0.0, 0.0]

b = 1.0 # 1.0
xb = [0.0, 0.0, np.pi, 0.0]

ua = [0.0]
ub = [0.0]

Refsol = True
if Refsol:
    path = 'E:\Yifan_Xue\DA\Data\example1(k,0.1,5)_t_0.9'
    refsol_x_place = open(path + '\\x_refsol.plk', 'rb')
    refsol_x = pickle.load(refsol_x_place)
    refsol_x_place.close()
    refsol_u_place = open(path + '\\u_refsol.plk', 'rb')
    refsol_u = pickle.load(refsol_u_place)
    refsol_u_place.close()
    xa = refsol_x[0]
    ua = refsol_u[0]
    b = 0.1

par = [1.5]
dt_sim = 0.01
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, dt_sim=dt_sim, su=4, sx=4, kx=2, use_chains=False, k=par, sol_steps=100, maxIt=4)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))


import matplotlib.pyplot as plt

plot_ori_Ref = True
if plot_ori_Ref:
    plt.figure(1)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    t = S.sim_data[0]
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


    plt.figure(3)
    plt.plot(range(len(S.k_list)), S.k_list, '.')
    plt.show()
    print len(S.k_list)



plot_Bro = False
if plot_Bro:
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    i, = np.where(S.sim_data_tt == 0.9)
    t = S.sim_data[0][i[0]:]
    x1 = S.sim_data[1][i[0]:, 0]
    x2 = S.sim_data[1][i[0]:, 1]
    x3 = S.sim_data[1][i[0]:, 2]
    x4 = S.sim_data[1][i[0]:, 3]
    u1 = S.sim_data[2][i[0]:, 0]

    plt.sca(ax1)
    plt.plot(t, x1, 'g')
    plt.title(r'$\alpha$')
    plt.xlabel('t')
    plt.ylabel('x1')

    plt.sca(ax2)
    plt.plot(t, x2, 'r')
    plt.xlabel('t')
    plt.ylabel('x2')

    plt.sca(ax3)
    plt.plot(t, x3, 'r')
    plt.xlabel('t')
    plt.ylabel('x3')

    plt.sca(ax4)
    plt.plot(t, x4, 'r')
    plt.xlabel('t')
    plt.ylabel('x4')

    plt.figure(2)
    plt.plot(t, u1, 'b')
    plt.xlabel('t')
    plt.ylabel(r'$u_{1}$')
    plt.show()

path = 'E:\Yifan_Xue\DA\Data\example1(k,0.1,5)_t_0.9'
save_res = False
if save_res:
    i, = np.where(S.sim_data_tt == 0.9)
    res_x_data = S.sim_data_xx[i[0]:]
    res_x_place = open(path+'\\x_refsol.plk', 'wb')
    pickle.dump(res_x_data, res_x_place)
    res_x_place.close()

    res_u_data = S.sim_data_uu[i[0]:]
    res_u_place = open(path+'\\u_refsol.plk', 'wb')
    pickle.dump(res_u_data, res_u_place)
    res_u_place.close()

    res_t_data = S.sim_data_tt[i[0]:]-0.9
    res_t_data[-1] = round(res_t_data[-1],5)
    res_t_place = open(path+'\\t_refsol.plk', 'wb')
    pickle.dump(res_t_data, res_t_place)
    res_t_place.close()
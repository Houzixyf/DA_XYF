from pytrajectory import ControlSystem, log
import numpy as np
import time
from sympy import cos, sin
from pytrajectory import penalty_expression as pe
import pickle


def f(x,u, par, evalconstr=False):
    # k, = par
    x1, x2, x3 = x       # system state variables
    u1, u2 = u                  # input variable


    ff = [  u1,
            u2,
            x2*u1-x1*u2,
        ]

    ff = [1 * eq for eq in ff]

    if evalconstr:
        res = 1*pe(k, 0.1, 5) #  pe(k, 0, 10)
        ff.append(res)
    return ff

a = 0.0
xa = [0, 0, 0]

b = 1.0
xb = [0, 0, 0.0]

ua = [0.0, 0.0]
ub = [0.0, 0.0]
par = [1.23]
S_time = 0.9
dt_sim = round((1.0 - S_time) / 10, 5)
dt_sim=dt_sim

path = 'E:\Yifan_Xue\DA\Codes\Data\with_Refsol_Brockett\Data_for_Brockett_e2_t_' + str(S_time)
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100,maxIt=5)
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], par))


save_res = False
if save_res:
    i, = np.where(S.sim_data_tt == S_time)
    res_x_data = S.sim_data_xx[i[0]:]
    res_x_place = open(path + '\\x_refsol.plk', 'wb')
    pickle.dump(res_x_data, res_x_place)
    res_x_place.close()

    res_u_data = S.sim_data_uu[i[0]:]
    res_u_place = open(path + '\\u_refsol.plk', 'wb')
    pickle.dump(res_u_data, res_u_place)
    res_u_place.close()

    res_t_data = S.sim_data_tt[i[0]:] - S_time
    res_t_data[-1] = round(res_t_data[-1], 5)
    res_t_place = open(path + '\\t_refsol.plk', 'wb')
    pickle.dump(res_t_data, res_t_place)
    res_t_place.close()

plot = True # without Refsol
if plot:
    import matplotlib.pyplot as plt
    i, = np.where(S.sim_data_tt == S_time)
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
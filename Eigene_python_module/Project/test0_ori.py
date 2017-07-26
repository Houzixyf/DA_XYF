"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
import numpy as np
from pytrajectory import penalty_expression as pe
import pickle
from pytrajectory.auxiliary import Container
log.console_handler.setLevel(10)


use_refsol = True # use refsol
Data = 'd:\\temp_data'
if use_refsol:
    refsol = Container()

    ref_for_xx = open(Data+'\\x_refsol.plk', 'rb')
    refsol.xx = pickle.load(ref_for_xx)
    ref_for_xx.close()
    ref_for_uu = open(Data+'\\u_refsol.plk', 'rb')
    refsol.uu = pickle.load(ref_for_uu)
    ref_for_uu.close()
    ref_for_tt = open(Data+'\\t_refsol.plk', 'rb')
    refsol.tt = pickle.load(ref_for_tt)
    refsol.tt[-1] = round(refsol.tt[-1],5)
    ref_for_tt.close()
    refsol.n_raise_spline_parts = 0


check_Brockett = False
if check_Brockett:
    che_for_xx = open(Data + '\\x.plk', 'rb')
    che_xx = pickle.load(che_for_xx)
    che_for_xx.close()
    che_for_uu = open(Data + '\\u.plk', 'rb')
    che_uu = pickle.load(che_for_uu)
    che_for_uu.close()
    che_for_tt = open(Data + '\\t.plk', 'rb')
    che_tt = pickle.load(che_for_tt)
    che_for_tt.close()
    print ('xa_old:{}'.format(che_xx))
    xa = che_xx
    print ('xa_new:{}'.format(xa))
    ua = che_uu
    # ua = [0.0]
    a = 0.0
    b = 0.01
    print('*********************')
    print('Begint the new loop')
    print('*********************')





# first, we define the function that returns the vectorfield
def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2 = x       # system state variables
    u1, = u                  # input variable


    ff = [  x2,
            u1,
        ]

    ff = [k * eq for eq in ff]

    if evalconstr:
        res = pe(k, .5, 10)# value of k in 0.5 and 10 is about 0
        ff.append(res)
    return ff


if 0:
    from matplotlib import pyplot as plt
    from ipHelp import IPS
    import sympy as sp
    kk = np.linspace(-20, 20)
    x = sp.Symbol('x')
    pefnc = sp.lambdify(x, pe(x, 0, 10), modules='numpy')
    #IPS()
    plt.plot(kk, pefnc(kk))
    plt.plot(kk, (kk-5)**2)
    plt.show()


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0]
xa = refsol.xx[0]

b = 1.0
xb = [1.0, 0.0]
b = 0.01

ua = [0.0]
ub = [0.0]
par = [1.0, 2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub,
                  su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100, dt_sim=0.01, refsol=refsol)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
save_refsol = False
if save_refsol:
    i, = np.where(S.sim_data_tt == 0.99)
    res_x = S.sim_data_xx[i[0]:]
    res_u = S.sim_data_uu[i[0]:]
    res_t = S.sim_data_tt[i[0]:] - 0.99
    save_res_x = open(Data + '\\x_refsol.plk', 'wb')
    pickle.dump(res_x, save_res_x)
    save_res_x.close()
    save_res_u = open(Data + '\\u_refsol.plk', 'wb')
    pickle.dump(res_u, save_res_u)
    save_res_u.close()
    save_res_t = open(Data + '\\t_refsol.plk', 'wb')
    pickle.dump(res_t, save_res_t)
    save_res_t.close()


print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))

# import matplotlib.pyplot as plt
# plt.figure(1)
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212)
#
# t = S.sim_data[0]
# x1 = S.sim_data[1][:, 0]
# x2 = S.sim_data[1][:, 1]
# u1 = S.sim_data[2][:, 0]
#
# plt.figure(1)
# plt.sca(ax1)
# plt.plot(t, x1, 'g')
# plt.title(r'$\alpha$')
# plt.xlabel('t')
# plt.ylabel(r'$x_{1}$')
#
# plt.sca(ax2)
# plt.plot(t, x2, 'r')
# plt.xlabel('t')
# plt.ylabel(r'$x_{2}$')
#
# plt.figure(2)
# plt.plot(t, u1, 'b')
# plt.xlabel('t')
# plt.ylabel(r'$u_{1}$')
# plt.show()

# plt.figure(3)
# plt.plot(range(len(S.k_list)), S.k_list)
# plt.show()
# print len(S.k_list)
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
import pickle
from pytrajectory.auxiliary import Container
log.console_handler.setLevel(10)

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
        res = 1 * pe(k, 0.1, 10)  # pe(k, 0.1, 15) -> k=11s
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

plot = False
T_time = []
SP = []
Time_SP = []
Reached_Accuracy = []
K_List = []
U_Umgebung = []
X1_Umgebung = []
X2_Umgebung = []
X3_Umgebung = []
X4_Umgebung = []
D = []

use_refsol = True # use refsol
Data = 'E:\Yifan_Xue\DA\Data\Data_for_Brockett_pe(k,0.1,15)_t_0.99'
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





for i in range(1):
    first_guess = None # {'seed': 5}

    delta_xx_1 = np.array([np.pi / 180, 0, 0, 0])
    delta_xx_2 = np.array([0, 0.01, 0, 0])
    delta_xx_3 = np.array([0, 0, np.pi / 180, 0])
    delta_xx_4 = np.array([0, 0, 0, 0.01])
    for d1 in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            for d3 in [-1, 0, 1]:
                for d4 in [-1, 0, 1]:
                    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
                        continue
                    x_change = (d1 * delta_xx_1 + d2 * delta_xx_2 + d3 * delta_xx_3 + d4 * delta_xx_4)

                    check_Brockett = True
                    if check_Brockett:
                        che_for_xx = open(Data+ '\\x.plk', 'rb')
                        che_xx = pickle.load(che_for_xx)
                        che_for_xx.close()
                        che_for_uu = open(Data +'\\u.plk', 'rb')
                        che_uu = pickle.load(che_for_uu)
                        che_for_uu.close()
                        che_for_tt = open(Data +'\\t.plk', 'rb')
                        che_tt = pickle.load(che_for_tt)
                        che_for_tt.close()
                        print ('xa_old:{}'.format(che_xx))
                        xa = che_xx + x_change
                        print ('xa_new:{}'.format(xa))
                        ua = che_uu
                        # ua = [0.0]
                        a = 0.0
                        b = 0.01
                        print('*********************')
                        print('Begint the new loop')
                        print('*********************')


                    par = [1.5]  # par must re-init
                    dt_sim = 0.001
                    # now we create our Trajectory object and alter some method parameters via the keyword arguments
                    S = ControlSystem(f, a, b, xa, xb, ua, ub, su=20, sx=10, kx=3, use_chains=False, k=par, sol_steps=100, dt_sim=dt_sim, first_guess=first_guess, maxIt=2, refsol=refsol)  # k must be a list,  k=par,  first_guess=first_guess,

                    T_start = time.time()
                    x, u, par = S.solve()
                    T_end = time.time()
                    print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))
                    T_time.append(T_end - T_start)
                    SP.append(S.nIt)
                    if S.reached_accuracy == True:
                        reached = 'True'
                    else:
                        reached = 'False'
                    Reached_Accuracy.append(reached)
                    K_List.append(S.eqs.sol[-1])
                    Time_SP.append([T_time[i], SP[i], Reached_Accuracy[i]])
                    u_list = S.sim_data[2]
                    x_list = S.sim_data[1]
                    U_Umgebung.append(u_list[:,0])
                    X1_Umgebung.append(x_list[:,0])
                    X2_Umgebung.append(x_list[:, 1])
                    X3_Umgebung.append(x_list[:, 2])
                    X4_Umgebung.append(x_list[:, 3])
                    D.append([d1, d2, d3, d4])
                    print ('Time to solve with seed-{}(s): {}'.format(i, T_time[-1]))


Save_in_Excel = True
from win32com.client import Dispatch
xl = Dispatch('Excel.Application')
xl.Quit()


if Save_in_Excel:
    import xlwt # save u in Excel

    workbook = xlwt.Workbook()  # encoding='utf-8'
    booksheet = workbook.add_sheet('U', cell_overwrite_ok=False)
    for i, row in enumerate(U_Umgebung):
        for j, col in enumerate(row):
            booksheet.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheet2 = workbook.add_sheet('D', cell_overwrite_ok=False )
    for i, row in enumerate(D):
        for j, col in enumerate(row):
            booksheet2.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetx_1 = workbook.add_sheet('X1', cell_overwrite_ok=False)
    for i, row in enumerate(X1_Umgebung):
        for j, col in enumerate(row):
            booksheetx_1.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetx_2 = workbook.add_sheet('X2', cell_overwrite_ok=False)
    for i, row in enumerate(X2_Umgebung):
        for j, col in enumerate(row):
            booksheetx_2.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetx_3 = workbook.add_sheet('X3', cell_overwrite_ok=False)
    for i, row in enumerate(X3_Umgebung):
        for j, col in enumerate(row):
            booksheetx_3.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetx_4 = workbook.add_sheet('X4', cell_overwrite_ok=False)
    for i, row in enumerate(X4_Umgebung):
        for j, col in enumerate(row):
            booksheetx_4.write(i, j, col)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetR = workbook.add_sheet('Reached_Accuracy', cell_overwrite_ok=False)
    for i, row in enumerate(Reached_Accuracy):
        booksheetR.write(i, 0, row)
    #workbook.save('d:\\U_Umgebung.xls')

    booksheetK = workbook.add_sheet('K', cell_overwrite_ok=False)
    for i, row in enumerate(K_List):
        booksheetK.write(i, 0, row)
    workbook.save('d:\\U_Umgebung.xls')




if 0: # plot
    if plot:
        import matplotlib.pyplot as plt

        t = S.sim_data[0]
        print t
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
print '\n'
print ('Time, Number of Iteration, Reached accuracy or not: {}').format(Time_SP)

save_sol_cx_cu = False  # save Coeffs cx100... and cu100...
if save_sol_cx_cu:
    sol_cx_cu = open('d:\\cx_cu.plk', 'wb')
    pickle.dump(S.eqs.solver.sol, sol_cx_cu)
    sol_cx_cu.close()

from IPython import embed as IPS
IPS()
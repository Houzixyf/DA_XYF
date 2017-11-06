# -*- coding: utf-8 -*-
# 此例用于研究 利用pytrajectory产生的u(t)在何种情况下是由规律的
# 当前例子是 x1和x3的初始值保持不变，分别为 0.0， -0.01， x2初始值从0.01减到0.003，每次相减0.001，在delta_xx_1中设置。




from pytrajectory import ControlSystem, log
import numpy as np
import time
from sympy import cos, sin
from pytrajectory import penalty_expression as pe

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
xa_start = [-0.1,0.1,0.1]

b = 1.0
xb = [0, 0, 0.0]

ua = [0.0, 0.0]
ub = [0.0, 0.0]
par = [1]


path = 'E:\Yifan_Xue\DA\Data\without_Refsol_Brockett\Brockett_e2\delta_x_0.01' # 此处需要更改
T_time = []
SP = []
Time_SP = []
Reached_Accuracy = []
K_List = []
U1_Umgebung = []
U2_Umgebung = []
X1_Umgebung = []
X2_Umgebung = []
X3_Umgebung = []
D = []

xa_start = [-0.1,0.1,0.1] # 此处需要更改         # if x1:-0.01 -> 0   (xa_start_new[0]<=0.001)
delta_xx_1 = np.array([0.01*1, 0, 0]) # 此处需要更改
delta_xx_2 = np.array([0, -0.01*1, 0])
# delta_xx_3 = np.array([0, 0, 0.1*1])

xa_start_new = xa_start
if 1:
    if 1:
        if 1:
            for d1 in range(int((0 - (-0.09)) / 0.01) + 1):
        # while(xa_start_new[1]>=0.003 and xa_start_new[0]<=0.04):# 此处需要更改
                x_change = d1*delta_xx_1
                xa = xa_start_new + x_change
                # xa_start_new = xa
                par = [1.0]
    # now we create our Trajectory object and alter some method parameters via the keyword arguments

                S = ControlSystem(f, a, b, xa, xb, ua, ub,
                      su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100, maxIt=5)  # k must be a list, dt=0.01
                T_start = time.time()
    # time to run the iteration
                x, u, par = S.solve()
                T_end = time.time()
                #print('x1(b)={}, x2(b)={}, x3(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[1][-1][2], S.sim_data[2][-1][0], S.eqs.sol[-1]))
                #print(S.eqs.sol)
                T_time.append(T_end - T_start)
                SP.append(S.nIt)
                if S.reached_accuracy:
                    reached = 'True'
                else:
                    reached = 'False'
                Reached_Accuracy.append(reached)
                K_List.append(S.eqs.sol[-1])
                Time_SP.append([T_time[0], SP[0], Reached_Accuracy[0]])
                u_list = S.sim_data[2]
                x_list = S.sim_data[1]
                U1_Umgebung.append(u_list[:, 0])
                U2_Umgebung.append(u_list[:, 1])
                X1_Umgebung.append(x_list[:, 0])
                X2_Umgebung.append(x_list[:, 1])
                X3_Umgebung.append(x_list[:, 2])


X_Umgebung = [X1_Umgebung,X2_Umgebung,X3_Umgebung]
U_Umgebung = [U1_Umgebung,U2_Umgebung]

print X1_Umgebung
# from IPython import embed as IPS
# IPS()

if 0:
    Save_in_Excel = True
    from win32com.client import Dispatch
    xl = Dispatch('Excel.Application')
    xl.Quit()


    if Save_in_Excel:
        import xlwt # save u in Excel

        workbook = xlwt.Workbook()  # encoding='utf-8'
        booksheetU1 = workbook.add_sheet('U1', cell_overwrite_ok=False)
        for i, row in enumerate(U1_Umgebung):
            for j, col in enumerate(row):
                booksheetU1.write(i, j, col)

        booksheetU2 = workbook.add_sheet('U2', cell_overwrite_ok=False)
        for i, row in enumerate(U2_Umgebung):
            for j, col in enumerate(row):
                booksheetU2.write(i, j, col)

        booksheet2 = workbook.add_sheet('D', cell_overwrite_ok=False )
        for i, row in enumerate(D):
            for j, col in enumerate(row):
                booksheet2.write(i, j, col)

        booksheetx_1 = workbook.add_sheet('X1', cell_overwrite_ok=False)
        for i, row in enumerate(X1_Umgebung):
            for j, col in enumerate(row):
                booksheetx_1.write(i, j, col)

        booksheetx_2 = workbook.add_sheet('X2', cell_overwrite_ok=False)
        for i, row in enumerate(X2_Umgebung):
            for j, col in enumerate(row):
                booksheetx_2.write(i, j, col)

        booksheetx_3 = workbook.add_sheet('X3', cell_overwrite_ok=False)
        for i, row in enumerate(X3_Umgebung):
            for j, col in enumerate(row):
                booksheetx_3.write(i, j, col)

        booksheetR = workbook.add_sheet('Reached_Accuracy', cell_overwrite_ok=False)
        for i, row in enumerate(Reached_Accuracy):
            booksheetR.write(i, 0, row)

        booksheetK = workbook.add_sheet('K', cell_overwrite_ok=False)
        for i, row in enumerate(K_List):
            booksheetK.write(i, 0, row)

        booksheetSP = workbook.add_sheet('SP', cell_overwrite_ok=False)
        for i, row in enumerate(SP):
            booksheetSP.write(i, 0, row)

        booksheetT_time = workbook.add_sheet('T_time', cell_overwrite_ok=False)
        for i, row in enumerate(T_time):
            booksheetT_time.write(i, 0, row)

        workbook.save('d:\\U_Umgebung.xls')

plot=True
if plot:
    import matplotlib.pyplot as plt
    import matplotlib
    #import seaborn as sns
    from matplotlib import rc

    #a = sns.color_palette("husl", n_colors=10)
    #sns.set_palette(a)
    matplotlib.rc('xtick', labelsize=50)
    matplotlib.rc('ytick', labelsize=50)
    matplotlib.rcParams.update({'font.size': 50})
    matplotlib.rcParams['xtick.major.pad'] = 12
    matplotlib.rcParams['ytick.major.pad'] = 12  # default = 3.5, distance to major tick label in points
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    t = S.sim_data[0]
    plt.figure(1)
    nx = 2
    if len(xa) % 2 == 0:  # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for i in [0,1]:
        plt.subplot(1, nx, i + 1)  # ax1 =
        plt.grid()
        for j in range(len(X1_Umgebung)):
            if 1:# j in [1,2,3,5,8,9]:
                plt.plot(t, X_Umgebung[i][j], linewidth=3.0, label=r'$x_{}|0={}$'.format(1,X1_Umgebung[j][0]))
                plt.grid(True)
            else:
                plt.plot(t, X_Umgebung[i][j], linewidth=3.0)
                plt.grid(True)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=50)
        plt.xlim(xmax=2)
        plt.ylabel(r'$x_{}$'.format(i + 1), fontsize=50)
        plt.legend(loc='best', fontsize=35)
        #import matplotlib.ticker as mtick
        #ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))#科学计数法
    if 1:
        plt.figure(3)
        plt.subplot(1, nx, 1)
        # plt.subplot(1, nx, i + 1)  # ax1 =
        plt.grid()
        for j in range(len(X1_Umgebung)):
            if 1:#j in [1,2,3,5,8,9]:
                plt.plot(t, X_Umgebung[2][j], linewidth=3.0, label=r'$x_{}|0={}$'.format(1,X1_Umgebung[j][0]))
                plt.grid(True)
            else:
                plt.plot(t, X_Umgebung[2][j], linewidth=3.0)
                plt.grid(True)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=50)
        plt.xlim(xmax=2)
        plt.ylabel(r'$x_{}$'.format(3), fontsize=50)
        plt.legend(loc='upper right', fontsize=35)
        axes = plt.subplot(1,2,2)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

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
        plt.grid()
        for j in range(len(U1_Umgebung)):
            if j in [1,2,3,5,8,9]:
                plt.plot(t, U_Umgebung[i][j], linewidth=3.0, label=r'$x_{}|0={}$'.format(1,X1_Umgebung[j][0]))
                plt.grid(True)
            else:
                plt.plot(t, U_Umgebung[i][j], linewidth=3.0)
                plt.grid(True)
        #     plt.title()
        plt.xlabel(r'$t(s)$', fontsize=50)
        plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=50)
        plt.legend(loc='best', fontsize=35)

    # plt.figure(3)
    # plt.plot(range(len(S.k_list)), S.k_list, '.', markersize=8, linewidth=3.0)
    # plt.xlabel(r'$Iteration-Mal$', fontsize=46)
    # plt.ylabel('$k$', fontsize=46)
    plt.grid(True)
    plt.show()
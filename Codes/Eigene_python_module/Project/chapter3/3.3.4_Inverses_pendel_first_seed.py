# -*- coding: utf-8 -*-
"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
import numpy as np
from numpy import pi
from sympy import cos, sin
import time
log.console_handler.setLevel(10)


# first, we define the function that returns the vectorfield


# 加上pe()的好处是最终的结果中，输入u的值会比较小，说明用一个较小的输入就可以控制系统； 但是 k 的范围依然需要手动确定，并不是给任意一个 k 都可以得到结果。



def f(x,u,par):
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable
    k = par[0]

    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([x2,
                    u1,
                    x4,
                   g / l * s + 1 / l * c * u1])# (1/l)*(g*sin(x3)+u1*cos(x3))
    return [k * eq for eq in ff]


a = 0.0
xa = [0.0, 0.0, 0.0, 0.0] # np.pi

b = 1.0
xb = [0.0, 0.0, np.pi, 0.0]

ua = [0.0]
ub = [0.0]



plot = False
T_time=[]
SP = []
Time_SP = []
Reached_Accuracy = []
K_List = []
FS_List = []
for i in range(100):
    first_guess = {'seed': i}
    par = [1.23]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
    S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par, dt_sim = 0.01, first_guess=first_guess, maxIt=6)  # k must be a list,  k=par,

    T_start = time.time()
    x, u, par = S.solve()
    T_end = time.time()
    print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))
    T_time.append(T_end-T_start)
    SP.append(S.nIt)
    if S.reached_accuracy:
        reached = 'True'
    else:
        reached = 'False'
    Reached_Accuracy.append(reached)
    K_List.append(S.k_list)
    Time_SP.append([T_time[i],SP[i],Reached_Accuracy[i]])
    FS = S.FS[0]
    FS_List.append(FS)
    print ('Time to solve with seed-{}(s): {}'.format(i, T_time[-1]))

Save_in_Excel = True
path = 'E:\Yifan_Xue\DA\Data\example0_time'
if 1:
    if Save_in_Excel:
        import xlwt

        workbook = xlwt.Workbook()  # encoding='utf-8'
        booksheet = workbook.add_sheet('T_time', cell_overwrite_ok=False)
        for i, row in enumerate(T_time):
            booksheet.write(i, 0, row)

        booksheet_SP = workbook.add_sheet('SP', cell_overwrite_ok=False )
        for i, row in enumerate(SP):
            booksheet_SP.write(i, 0, row)

        booksheet_RA = workbook.add_sheet('Reached_Accuracy', cell_overwrite_ok=False)
        for i, row in enumerate(Reached_Accuracy):
            booksheet_RA.write(i, 0, row)

        booksheet_K = workbook.add_sheet('K', cell_overwrite_ok=False)
        for i, row in enumerate(K_List):
            for j, col in enumerate(row):
                booksheet_K.write(i, j, float(col))

        booksheet_FS = workbook.add_sheet('First_Guess', cell_overwrite_ok=False)
        for i, row in enumerate(FS_List):
            for j, col in enumerate(row):
                booksheet_FS.write(i, j, col)

        workbook.save(path + '\\first_guess_reverse.xls')

print '\n'
print ('Time, Number of Iteration, Reached accuracy or not: {}').format(Time_SP)
from IPython import embed as IPS
IPS()
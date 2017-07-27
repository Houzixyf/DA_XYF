"""
inverted pendulum, Brockett


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
from pytrajectory import penalty_expression as pe
from sympy import cos, sin
import pickle
import numpy as np
import time
from pytrajectory.auxiliary import Container

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




# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, 0.0, 0.0]

b = 1.0
xb = [0.0, 0.0, np.pi, 0.0]
par = [1.5]

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


path = 'E:\Yifan_Xue\DA\Data\example1(k,0.1,5)_t_0.9'
S_time = 0.9
use_refsol = True
if use_refsol:
    refsol_x_place = open(path+'\\x_refsol.plk', 'rb')
    refsol_x = pickle.load(refsol_x_place)
    refsol_x_place.close()

    refsol_u_place = open(path+'\\u_refsol.plk', 'rb')
    refsol_u = pickle.load(refsol_u_place)
    refsol_u_place.close()

    refsol_t_place = open(path+'\\t_refsol.plk', 'rb')
    refsol_t = pickle.load(refsol_t_place)
    refsol_t_place.close()

    b = round(1.0 - S_time, 5)
    xa_start = refsol_x[0]
    # xb = refsol_x[-1]
    ua = refsol_u[0]
    ub = 0
    # ub = refsol_u[-1]

    Refsol = Container()
    Refsol.tt = refsol_t
    Refsol.xx = refsol_x
    Refsol.uu = refsol_u
    Refsol.n_raise_spline_parts = 0


delta_xx_1 = np.array([0.01, 0, 0, 0])
delta_xx_2 = np.array([0, 0.15, 0, 0])
delta_xx_3 = np.array([0, 0, np.pi / 36, 0])
delta_xx_4 = np.array([0, 0, 0, np.pi/10])

for d1 in [-1, 0, 1]:
    for d2 in [-1, 0, 1]:
        for d3 in [-1, 0, 1]:
            for d4 in [-1, 0, 1]:
                x_change = (d1 * delta_xx_1 + d2 * delta_xx_2 + d3 * delta_xx_3 + d4 * delta_xx_4)
                xa = xa_start + x_change
                par = [1.5]
                dt_sim = 0.01
# now we create our Trajectory object and alter some method parameters via the keyword arguments
                S = ControlSystem(f, a, b, xa, xb, ua, ub, dt_sim=dt_sim, su=4, sx=4, kx=2, use_chains=False, k=par, sol_steps=100, maxIt=4)  # k must be a list
                T_start = time.time()
                x, u, par = S.solve()
                T_end = time.time()
                print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))
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
                U_Umgebung.append(u_list[:, 0])
                X1_Umgebung.append(x_list[:, 0])
                X2_Umgebung.append(x_list[:, 1])
                X3_Umgebung.append(x_list[:, 0])
                X4_Umgebung.append(x_list[:, 1])
                D.append([d1, d2, d3, d4])




if 1:
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

        booksheetx_4 = workbook.add_sheet('X4', cell_overwrite_ok=False)
        for i, row in enumerate(X4_Umgebung):
            for j, col in enumerate(row):
                booksheetx_4.write(i, j, col)

        booksheetR = workbook.add_sheet('Reached_Accuracy', cell_overwrite_ok=False)
        for i, row in enumerate(Reached_Accuracy):
            booksheetR.write(i, 0, row)

        booksheetK = workbook.add_sheet('K', cell_overwrite_ok=False)
        for i, row in enumerate(K_List):
            booksheetK.write(i, 0, row)
        workbook.save(path+'\\U_Umgebung.xls')

from IPython import embed as IPS
IPS()
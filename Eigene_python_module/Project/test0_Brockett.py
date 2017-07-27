'''
Doppelintegrator, Brockett
'''

# import all we need for solving the problem
from pytrajectory import ControlSystem
from pytrajectory import penalty_expression as pe
import pickle
import numpy as np
import time
from pytrajectory.auxiliary import Container
# the next imports are necessary for the visualisatoin of the system


# first, we define the function that returns the vectorfield
def f(x, u, par, evalconstr=True):
    x1, x2= x  # system variables
    u1, = u  # input variable
    k = par[0]

    # this is the vectorfield
    ff = [k*x2,
          k*u1]

    if evalconstr:
        res = 1 * pe(k, 0.1, 10)
        ff.append(res)
    return ff


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0]

b = 1.0
xb = [1.0, 0.0]
par = [1.5]

T_time = []
SP = []
Time_SP = []
Reached_Accuracy = []
K_List = []
U_Umgebung = []
X1_Umgebung = []
X2_Umgebung = []
D = []

path = 'E:\Yifan_Xue\DA\Data\Doppelintegrator_t_0.99'
use_refsol = True
if use_refsol:
    refsol_x_place = open(path+'\\res_x.pkl', 'rb')
    refsol_x = pickle.load(refsol_x_place)
    refsol_x_place.close()

    refsol_u_place = open(path+'\\res_u.pkl', 'rb')
    refsol_u = pickle.load(refsol_u_place)
    refsol_u_place.close()

    refsol_t_place = open(path+'\\res_t.pkl', 'rb')
    refsol_t = pickle.load(refsol_t_place)
    refsol_t_place.close()
    b = 0.01
    xa_start = refsol_x[0]
    ua = refsol_u[0]
    ub = 0

    Refsol = Container()
    Refsol.tt = refsol_t
    Refsol.xx = refsol_x
    Refsol.uu = refsol_u
    Refsol.n_raise_spline_parts = 0



# first_guess = {'seed':1} # {'seed':1}
# now we create our Trajectory object and alter some method parameters via the keyword arguments
delta_xx_1 = np.array([0.1, 0])
delta_xx_2 = np.array([0, 0.01])

for d1 in [-1, 0, 1]:
    for d2 in [-1, 0, 1]:
        x_change = (d1 * delta_xx_1 + d2 * delta_xx_2)
        xa = xa_start + x_change
        par = [1.5]
        dt_sim = 0.001
        S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par, dt_sim=dt_sim, first_guess=None, refsol=Refsol)  # k must be a list
        T_start = time.time()
        x, u, par = S.solve()
        T_end = time.time()
        print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1],
                                                         S.sim_data[2][-1][0], S.eqs.sol[-1]))
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
        D.append([d1, d2])


plot = False # using Refsol
if plot:
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    t = S.sim_data[0]
    x1 = S.sim_data[1][:, 0]
    x2 = S.sim_data[1][:, 1]
    u1 = S.sim_data[2][:, 0]

    plt.sca(ax1)
    plt.plot(t, x1, 'g')
    plt.title(r'$\alpha$')
    plt.xlabel('t')
    plt.ylabel('x1')

    plt.sca(ax2)
    plt.plot(t,x2, 'r')
    plt.xlabel('t')
    plt.ylabel('x2')

    plt.figure(2)
    plt.plot(t, u1, 'b')
    plt.xlabel('t')
    plt.ylabel(r'$u_{1}$')
    plt.show()

plot = False # without Refsol
if plot:
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    i, = np.where(S.sim_data_tt == 0.9)
    t = S.sim_data[0][i[0]:]
    x1 = S.sim_data[1][i[0]:, 0]
    x2 = S.sim_data[1][i[0]:, 1]
    u1 = S.sim_data[2][i[0]:, 0]

    plt.sca(ax1)
    plt.plot(t, x1, 'g')
    plt.title(r'$\alpha$')
    plt.xlabel('t')
    plt.ylabel('x1')

    plt.sca(ax2)
    plt.plot(t,x2, 'r')
    plt.xlabel('t')
    plt.ylabel('x2')

    plt.figure(2)
    plt.plot(t, u1, 'b')
    plt.xlabel('t')
    plt.ylabel(r'$u_{1}$')
    plt.show()

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

        booksheetR = workbook.add_sheet('Reached_Accuracy', cell_overwrite_ok=False)
        for i, row in enumerate(Reached_Accuracy):
            booksheetR.write(i, 0, row)

        booksheetK = workbook.add_sheet('K', cell_overwrite_ok=False)
        for i, row in enumerate(K_List):
            booksheetK.write(i, 0, row)
        workbook.save(path+'\\U_Umgebung.xls')

from IPython import embed as IPS
IPS()
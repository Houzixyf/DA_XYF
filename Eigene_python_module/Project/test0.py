'''
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.
'''

# import all we need for solving the problem
from pytrajectory import ControlSystem
from pytrajectory import penalty_expression as pe
import pickle
import numpy as np
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
        res = 0 * pe(k, 0.1, 10)
        ff.append(res)
    return ff


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0]

b = 1.0
xb = [1.0, 0.0]

ua = [0.0]
ub = [0.0]
par = [1.5]

use_refsol = False
if use_refsol:
    refsol_x_place = open('d://res_x.pkl', 'rb')
    refsol_x = pickle.load(refsol_x_place)
    refsol_x_place.close()

    refsol_u_place = open('d://res_u.pkl', 'rb')
    refsol_u = pickle.load(refsol_u_place)
    refsol_u_place.close()

    refsol_t_place = open('d://res_t.pkl', 'rb')
    refsol_t = pickle.load(refsol_t_place)
    refsol_t_place.close()
    b = 0.1
    xa = refsol_x[0]
    ua = refsol_u[0]

    Refsol = Container()
    Refsol.tt = refsol_t
    Refsol.xx = refsol_x
    Refsol.uu = refsol_u




# first_guess = {'seed':1} # {'seed':1}
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par, first_guess=None, refsol=None)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))
from IPython import embed as IPS
# IPS()

save_res = False
if save_res:
    i, = np.where(S.sim_data_tt == 0.9)
    res_x_data = S.sim_data_xx[i[0]:]
    res_x_place = open('d://res_x.pkl', 'wb')
    pickle.dump(res_x_data, res_x_place)
    res_x_place.close()

    res_u_data = S.sim_data_uu[i[0]:]
    res_u_place = open('d://res_u.pkl', 'wb')
    pickle.dump(res_u_data, res_u_place)
    res_u_place.close()

    res_t_data = S.sim_data_tt[i[0]:]-0.9
    res_t_data[-1] = round(res_t_data[-1],5)
    res_t_place = open('d://res_t.pkl', 'wb')
    pickle.dump(res_t_data, res_t_place)
    res_t_place.close()







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

plot = True # without Refsol
if plot:
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    i, = np.where(S.sim_data_tt==0.9)
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


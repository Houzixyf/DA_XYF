# translation of the inverted pendulum

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
from sympy import sin, cos
import numpy as np
from numpy import pi
# define the function that returns the vectorfield
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

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0]

b = 1.0
xb = [0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

par = [1.23]
#con = {0 : [-0.2, 0.4]}
# con = {'par[0]' : [0.1, 10]}
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, maxIt=10, constraints=None, k=par, dt_sim=0.001) #constraints=con,
# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.sim_data[-1][0]))

print S.eqs.solver.sol
S.eqs.solver.sol.tolist()

# import pickle
# sol_for_k = open('d:\\sol.plk', 'wb')
# pickle.dump(S.eqs.solver.sol,sol_for_k)
# sol_for_k.close()



plot = True # using Refsol
if plot:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import rc

    matplotlib.rc('xtick', labelsize=58)
    matplotlib.rc('ytick', labelsize=58)
    matplotlib.rcParams.update({'font.size': 58})
    matplotlib.rcParams['xtick.major.pad'] = 12
    matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    t = S.sim_data[0]
    x1 = S.sim_data[1][:, 0]
    x2 = S.sim_data[1][:, 1]
    u1 = S.sim_data[2][:, 0]

    plt.figure(1)
    plt.grid()
    nx = 2
    if len(xa) % 2 == 0:  # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for i in ax:
        plt.subplot(mx, nx, i + 1)
        plt.plot(t, S.sim_data[1][:, i], linewidth=3.0)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=56)
        plt.grid()
        if i == 0:
            plt.ylabel(r'$x_{}(m)$'.format(i + 1), fontsize=56)
        if i == 1:
            plt.ylabel(r'$x_{}(m/s)$'.format(i + 1), fontsize=56)
        if i == 2:
            plt.ylabel(r'$x_{}(rad)$'.format(i + 1), fontsize=56)
        if i == 3:
            plt.ylabel(r'$x_{}(rad/s)$'.format(i + 1), fontsize=56)

    plt.figure(2)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    plt.sca(ax1)
    plt.plot(t, u1, linewidth=3.0)
    plt.xlabel(r'$t(s)$', fontsize=58)
    plt.ylabel(r'$u(m/s^2)$', fontsize=58)
    plt.grid()
    plt.sca(ax2)
    plt.plot(range(len(S.k_list)), S.k_list, '.',markersize = 10, linewidth=3.0)
    plt.xlabel(r'$Iteration$', fontsize=58)
    plt.ylabel('$k$', fontsize=58)
    plt.grid()
    plt.show()

    ## for plot k and sk (constraints)
    # plt.figure(3)
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # plt.sca(ax1)
    # plt.plot(range(len(S.sk_list)), S.sk_list, '.', markersize = 15, color='b', linewidth=3.0)
    # plt.xlabel(r'$Iteration-Mal$', fontsize=46)
    # plt.ylabel('$sk$', fontsize=46)
    # plt.grid()
    # plt.sca(ax2)
    # plt.plot(range(len(S.k_list)), S.k_list, '.', color='b', markersize = 15, linewidth=3.0)
    # plt.xlabel(r'$Iteration-Mal$', fontsize=46)
    # plt.ylabel('$k$', fontsize=46)
    # plt.grid()
    # plt.show()

# save free coefficients for get_guess in collocation
ip = False #(if False, then in collocation_get_guess, cx_cu True )
if ip:
    import pickle
    sol_for_k = open('d:\\sol_test0.plk', 'wb')
    pickle.dump(S.eqs.solver.sol,sol_for_k)
    sol_for_k.close()
print (S.eqs.solver.sol)
print (S.k_list)
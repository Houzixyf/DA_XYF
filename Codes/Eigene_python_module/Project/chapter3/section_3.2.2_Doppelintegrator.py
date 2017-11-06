# -*- coding: utf-8 -*-
from pytrajectory import ControlSystem
def f(x, u, par):
    x1, x2 = x  # system variables
    u1, = u  # input variable
    k = par[0]
    # this is the vectorfield
    ff = [k*x2,
          k*u1]
    return ff

a = 0.0
xa = [0.0, 0.0]

b = 1.0
xb = [1.0, 0.0]

ua = [0.0]
ub = [0.0]

par = [1.23]

con = {'par[0]' : [0.0, 5]}
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, first_guess=None, maxIt=10, par=par, constraints=con)  # k must be a list

# time to run the iteration
x, u, par= S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))
print (S.k_list)
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
        plt.xlabel(r'$t(s)$', fontsize=58)
        plt.grid()
        if i == 0:
            plt.ylabel(r'$x_{}(m)$'.format(i + 1), fontsize=58)
        if i == 1:
            plt.ylabel(r'$x_{}(m/s)$'.format(i + 1), fontsize=58)
        if i == 2:
            plt.ylabel(r'$x_{}(rad)$'.format(i + 1), fontsize=58)
        if i == 3:
            plt.ylabel(r'$x_{}(rad/s)$'.format(i + 1), fontsize=58)

    plt.figure(2)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(t, u1, linewidth=3.0)
    plt.xlabel(r'$t(s)$', fontsize=58)
    plt.ylabel(r'$u(m/s^2)$', fontsize=58)
    plt.grid()
    plt.sca(ax2)
    plt.plot(range(len(S.k_list)), S.k_list, '.',markersize = 15, linewidth=3.0)
    plt.xlabel(r'$Iteration$', fontsize=58)
    plt.ylabel('$k$', fontsize=58)
    plt.grid()
    plt.show()

    #for plot k and sk (constraints)
    plt.figure(3)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(range(len(S.sk_list)), S.sk_list, '.', markersize = 15, color='b', linewidth=3.0)
    plt.xlabel(r'$Iteration$', fontsize=58)
    plt.ylabel('$\kappa$', fontsize=58)
    plt.grid()
    plt.sca(ax2)
    plt.plot(range(len(S.k_list)), S.k_list, '.', color='b', markersize = 15, linewidth=3.0)
    plt.xlabel(r'$Iteration$', fontsize=58)
    plt.ylim(0,30)
    plt.ylabel('$k$', fontsize=58)
    plt.grid()
    plt.show()


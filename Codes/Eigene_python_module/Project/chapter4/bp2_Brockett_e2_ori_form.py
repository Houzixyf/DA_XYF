from pytrajectory import ControlSystem
import math
# the next imports are necessary for the visualisatoin of the system


# first, we define the function that returns the vectorfield
def f(x, u, par):
    x1, x2, x3= x  # system variables
    u1, u2= u  # input variable
    k = 1# par[0]

    # this is the vectorfield
    ff = [k*u1,
          k*u2,
          k*(x2*u1-x1*u2)]
    return ff
    # return [k * eq for eq in ff]


# then we specify all boundary conditions
a = 0.0
xa = [1.0,1.0,1.0]

b = 1.0
xb = [0.0,0.0,0.0]# math.e

ua = [0.0, 0.0]
ub = [0.0, 0.0]
par = [1.23,2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], par))

# plot

plot = 1
if plot:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import rc

    matplotlib.rc('xtick', labelsize=44)
    matplotlib.rc('ytick', labelsize=44)
    matplotlib.rcParams.update({'font.size': 44})
    matplotlib.rcParams['xtick.major.pad'] = 12
    matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    t = S.sim_data[0]
    plt.figure(1)
    nx = 2
    if len(xa) % 2 == 0: # a.size
        mx = len(xa) / nx
    else:
        mx = len(xa) / nx + 1

    ax = xrange(len(xa))

    for i in ax:
        plt.subplot(mx,nx,i+1)
        plt.grid()
        plt.plot(t,S.sim_data[1][:, i], linewidth=3.0)
        # plt.title()
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$x_{}$'.format(i+1), fontsize=46)

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
        plt.plot(t, S.sim_data[2][:, i], linewidth=3.0)
    #     plt.title()
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$u_{}$'.format(i + 1), fontsize=46)

    # plt.figure(3)
    # plt.plot(range(len(S.k_list)), S.k_list, '.', markersize = 15, linewidth=3.0)
    # plt.xlabel(r'$Iteration-Mal$', fontsize=46)
    # plt.ylabel('$k$', fontsize=46)
    # plt.grid()
    # plt.show()

    # print len(S.k_list)

    plt.show()
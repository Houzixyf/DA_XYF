"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
import numpy as np
from pytrajectory import penalty_expression as pe

log.console_handler.setLevel(10)


# first, we define the function that returns the vectorfield
def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2 = x       # system state variables
    u1, = u                  # input variable


    ff = [  x2,
            u1,
        ]

    # ff = [k * eq for eq in ff]

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

b = 1.0
xb = [1.0, 0.0]

ua = [0.0]
ub = [0.0]
par = [1.0, 2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub,
                  su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))

import matplotlib.pyplot as plt
plt.figure(1)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

t = S.sim_data[0]
x1 = S.sim_data[1][:, 0]
x2 = S.sim_data[1][:, 1]
u1 = S.sim_data[2][:, 0]

plt.figure(1)
plt.sca(ax1)
plt.plot(t, x1, 'g')
plt.title(r'$\alpha$')
plt.xlabel('t')
plt.ylabel(r'$x_{1}$')

plt.sca(ax2)
plt.plot(t, x2, 'r')
plt.xlabel('t')
plt.ylabel(r'$x_{2}$')

plt.figure(2)
plt.plot(t, u1, 'b')
plt.xlabel('t')
plt.ylabel(r'$u_{1}$')
plt.show()

plt.figure(3)
plt.plot(range(len(S.k_list)), S.k_list)
plt.show()
print len(S.k_list)
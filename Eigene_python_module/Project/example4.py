"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.


This version is used to investigate the influence of an additional free parameter.
"""

# import all we need for solving the problem
from pytrajectory import ControlSystem, log
import numpy as np
from sympy import cos, sin
from pytrajectory import penalty_expression as pe

log.console_handler.setLevel(10)


# first, we define the function that returns the vectorfield






def f(x,u, par, evalconstr=True):
    k, = par
    x1, x2, x3, x4 = x  # state variables
    u1, = u  # input variable

    e = 0.9  # inertia coupling

    s = sin(x3)
    c = cos(x3)

    ff = [         x2,
                   u1,
                   x4,
                   -e * x2 ** 2 * s - (1 + e * c) * u1
                   ]


    if evalconstr:
            res = pe(k, 0.1, 5) #  pe(k, 0, 10)
            ff.append(res)

    return [k * eq for eq in ff]


xa = [  0.0,
        0.0,
        0.4*np.pi,
        0.0]

xb = [  0.2*np.pi,
        0.0,
        0.2*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

a = 0.0
b = 1.8
par = [1.5]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub,
                  su=20, sx=10, kx=3, use_chains=True, k=par, sol_steps=100)  # k must be a list,  k=par,

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.eqs.sol[-1]))


import matplotlib.pyplot as plt
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
    plt.plot(t,S.sim_data[1][:, i])
    # plt.title()
    plt.xlabel('t')
    plt.ylabel(r'$x_{}$'.format(i+1))

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
    plt.plot(t, S.sim_data[2][:, i])
#     plt.title()
    plt.xlabel('t')
    plt.ylabel(r'$u_{}$'.format(i + 1))

plt.show()
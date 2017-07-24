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
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5     # length of the pendulum rod (distance to center of mass)
    g = 9.81    # gravitational acceleration

    s = sin(x3)
    c = cos(x3)

    ff = [  x2,
            u1,
            x4,
            -(1 / l) * (g * sin(x3) + u1 * cos(x3)) # -g/l*s - 1/l*c*u1
        ]

    ff = [k * eq for eq in ff]

    if evalconstr:
        res = 1*pe(k, 0.1, 10) #  pe(k, 0, 10)
        ff.append(res)
    return ff

# def f(x, u):
#
#     x1, x2, x3, x4 = x       # system state variables
#     u1, = u                  # input variable
#
#     l = 0.5     # length of the pendulum rod (distance to center of mass)
#     g = 9.81    # gravitational acceleration
#
#     s = sin(x3)
#     c = cos(x3)
#
#     ff = [  x2,
#             u1,
#             x4,
#             -(1 / l) * (g * sin(x3) + u1 * cos(x3)) # -g/l*s - 1/l*c*u1
#         ]
#
#     # ff = [k * eq for eq in ff]
#
#     return ff


if 0:
    from matplotlib import pyplot as plt
    from ipHelp import IPS
    import sympy as sp
    kk = np.linspace(0, 15)
    x = sp.Symbol('x')
    pefnc = sp.lambdify(x, pe(x, 0.1, 10), modules='numpy')
    #plt.semilogy(kk, pefnc(kk))
    plt.plot(kk, pefnc(kk))
    plt.show()
    raise SystemExit


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, 0.0, 0.0]

b = 1.0
xb = [1.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]
par = [1, 2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub,
                  su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100, maxIt=10 )  # k must be a list

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

# plt.figure(3)
# plt.plot(range(len(S.k_list)), S.k_list, '.')
# plt.show()
# print len(S.k_list)
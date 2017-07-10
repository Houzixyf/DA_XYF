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
    x1, x2, x3, x4, x5, x6 = x  # system state variables
    u1, u2 = u  # input variables

    # coordinates for the points in which the engines engage [m]
    l = 1.0
    h = 0.1

    g = 9.81  # graviational acceleration [m/s^2]
    M = 50.0  # mass of the aircraft [kg]
    J = 25.0  # moment of inertia about M [kg*m^2]

    alpha = 5 / 360.0 * 2 * np.pi  # deflection of the engines

    sa = sin(alpha)
    ca = cos(alpha)

    s = sin(x5)
    c = cos(x5)

    ff = [x2,
                   -s / M * (u1 + u2) + c / M * (u1 - u2) * sa,
                   x4,
                   -g + c / M * (u1 + u2) + s / M * (u1 - u2) * sa,
                   x6,
                   1 / J * (u1 - u2) * (l * ca + h * sa)]


    if evalconstr:
            res = pe(k, -5, 5) #  pe(k, 0, 10)
            ff.append(res)
    return ff


xa = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [10.0, 0.0, 5.0, 0.0, 0.0, 0.0]

# boundary values for the inputs
ua = [0.5*9.81*50.0/(cos(5/360.0*2*np.pi)), 0.5*9.81*50.0/(cos(5/360.0*2*np.pi))]
ub = [0.5*9.81*50.0/(cos(5/360.0*2*np.pi)), 0.5*9.81*50.0/(cos(5/360.0*2*np.pi))]

a = 0.0
b = 1.0
par = [1.5]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub,
                  su=2, sx=2, kx=2, use_chains=False, k=par, sol_steps=100)  # k must be a list

# time to run the iteration
S.solve()
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
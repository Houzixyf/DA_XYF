'''
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.
'''

# import all we need for solving the problem
from pytrajectory import ControlSystem
import numpy as np
from sympy import cos, sin
from numpy import pi

# the next imports are necessary for the visualisatoin of the system
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation


# first, we define the function that returns the vectorfield
def f(x,u, par):
    k = par[0]
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
            s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])

    return [k * eq for eq in ff]


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, 0.0, 0.0]

b = 1.0
xb = [1.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]
par = [10.23, 2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.sim_data[-1][0]))


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

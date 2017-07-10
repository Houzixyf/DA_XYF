# underactuated manipulator

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
import numpy as np
from sympy import cos, sin

# define the function that returns the vectorfield
def f(x,u,par):
    k = par[0]
    x1, x2, x3, x4  = x     # state variables
    u1, = u                 # input variable
    
    e = 0.9     # inertia coupling
    
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([         x2,
                            u1,
                            x4,
                    -e*x2**2*s-(1+e*c)*u1
                    ])
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 1.8 [s]
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
par = [10.23, 2.0]
a = 0
b = 1
# create trajectory object
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par)

# also alter some method parameters to increase performance

# run iteration
S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation
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
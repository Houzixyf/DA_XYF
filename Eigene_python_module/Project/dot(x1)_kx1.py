'''
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.
'''

# import all we need for solving the problem
from pytrajectory import ControlSystem

# the next imports are necessary for the visualisatoin of the system


# first, we define the function that returns the vectorfield
def f(x, u, par):
    x1, = x  # system variables
    u1, = u  # input variable
    k = par[0]

    # this is the vectorfield
    ff = [x1]

    return [k * eq for eq in ff]


# then we specify all boundary conditions
a = 0.0
xa = [1.0]

b = 1.0
xb = [1.0]

ua = [0.0]
ub = [0.0]
par = [1.23,2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, k=par)  # k must be a list now

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][0], S.sim_data[2][-1][0], S.sim_data[-1][0]))

# plot

import matplotlib.pyplot as plt
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

t = S.sim_data[0]
x1 = S.sim_data[1][:, 0]
u = S.sim_data[2][:, 0]

plt.sca(ax1)
plt.plot(t, x1, 'g')
plt.title(r'$\alpha$')
plt.xlabel('t')
plt.ylabel('x1')

plt.sca(ax2)
plt.plot(t, u, 'b')
plt.xlabel('t')
plt.ylabel(r'$u_{1}$')

plt.show()


# k = 4.18046741827e-21
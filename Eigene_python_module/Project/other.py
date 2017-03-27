from pytrajectory import ControlSystem
import math
# the next imports are necessary for the visualisatoin of the system


# first, we define the function that returns the vectorfield
def f(x, u, par):
    x1, x2 = x  # system variables
    u1, = u  # input variable
    k = par[0]

    # this is the vectorfield
    ff = [x2**2 + 1,
          u1]
    return [k * eq for eq in ff]


# then we specify all boundary conditions
a = 0.0
xa = [1.0, 0.0]

b = 1.0
xb = [4*math.e**2 - 16*math.e + 23, 2*math.e - 2] # math.e

ua = [1.0]
ub = [math.e]
par = [10.23, 2.0]
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=4, sx=4, kx=2, use_chains=False, k=par)  # k must be a list

# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.sim_data[-1][0]))

# plot

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


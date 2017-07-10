# translation of the inverted pendulum

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
from sympy import sin, cos
import numpy as np
from numpy import pi
# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([x2,
                    u1,
                    x4,
                   -g / l * s - 1 / l * c * u1])# (1/l)*(g*sin(x3)+u1*cos(x3))
    return [1 * eq for eq in ff]

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, 0.0, 0.0]

b = 1.0
xb = [1.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]
# con = {0 : [-0.1, 1.6]}
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = ControlSystem(f, a, b, xa, xb, ua, ub, su=2, sx=2, kx=2, use_chains=False, maxIt=10) #constraints=con,
# time to run the iteration
x, u, par = S.solve()
print('x1(b)={}, x2(b)={}, u(b)={}, k={}'.format(S.sim_data[1][-1][0], S.sim_data[1][-1][1], S.sim_data[2][-1][0], S.sim_data[-1][0]))

print S.eqs.solver.sol
S.eqs.solver.sol.tolist()

# import pickle
# sol_for_k = open('d:\\sol.plk', 'wb')
# pickle.dump(S.eqs.solver.sol,sol_for_k)
# sol_for_k.close()



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
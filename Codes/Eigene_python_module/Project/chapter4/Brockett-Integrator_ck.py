
# coding: utf-8

# Dieses Notebook untersucht den Brockett-Integrator
# 
# 
# \begin{align}
# %\label{eq_}
# \dot x_1 &= u_1\\
# \dot x_2 &= u_2\\
# \dot x_3 &= x_2 u_1 - x_1 u_2
# \end{align}
# 
# 
# 

# In[35]:

import time
time.ctime()


# ### Technische Abhängigkeiten:
# 
# 
# https://github.com/TUD-RST/symbtools
# 
# https://github.com/cknoll/displaytools
# 
# 

# In[36]:

# get_ipython().magic(u'load_ext displaytools3')

import sympy as sp
import numpy as np
import scipy as sc
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

import symbtools as st
from sympy.interactive import printing
from mpl_toolkits.mplot3d import Axes3D
printing.init_printing(1)


# In[37]:

xx = st.symb_vector("x1:4")
uu = st.symb_vector("u1:3")
zz = st.symb_vector("z1:4")
vv = st.symb_vector("v1:3")
st.make_global(xx, uu, zz, vv)


# ### Lässt sich das System in die *regular Form* (Khalil S. 564) überführen?
# → Involutivitätsuntersuchung

# In[38]:

b1 = sp.Matrix([1, 0, x2]) ##:
b2 = sp.Matrix([0, 1, -x1]) ##:
B = st.col_stack(b1, b2)


# Involutivitäts-Test

# In[39]:

st.involutivity_test(B, xx)


# → nicht involutiv.
# 
# 
# Das sieht man auch direkt

# In[40]:

st.lie_bracket(b1, b2, xx)


# ### Herleitung eines schaltenden Regelgesetzes, welches den Ursprung (vermutlich)  global asymtoptisch stabilisiert
#  * basiert auf Darstellung in Zylinderkoordinaten
# 
# Wdh.: Darstellung in Originalkoordinaten:
# 
# $
# \begin{align}
# %\label{eq_}
# \dot x_1 &= u_1\\
# \dot x_2 &= u_2\\
# \dot x_3 &= x_2 u_1 - x_1 u_2
# \end{align}
# $

# In[41]:

ff = B*uu ##:


# ### Herleitung: Systemdynamik in Zylinderkoordinaten:
# 
# 
# \begin{align}
# %\label{eq_}
# r &= z_1 =  \sqrt{x_1^2 + x_2^2} \\
# \varphi &= z_2 = \mathrm{arctan2}(x_2, x_1) \\
# z &= z_3 = x_3
# \end{align}
# 

# In[42]:

z1e = sp.sqrt(x1**2 + x2**2)
z2e = sp.atan2(x2, x1)
z3e = x3

zz_expr = sp.Matrix([z1e, z2e, z3e])

rplm1 = [(z1e, z1), (sp.expand(-ff[2]/z1**2), v2), (ff[2], -z1**2*v2)]

ffz = zz_expr.jacobian(xx)*ff ##:
ffz = ffz.subs(rplm1) ##:
rplm2 = [(ffz[0], v1)]
ffz = ffz.subs(rplm2) ##:


# In[43]:

vve = vv.subs(st.rev_tuple(rplm1+rplm2)) ##:


# Rücktranstformation des Eingangs

# In[44]:

M = vve.jacobian(uu).inverse_ADJ()
M.simplify()
M.subs(rplm1) ##:

# uue = M.subs(rplm1) * vv ##:

uue = M.subs(st.rev_tuple(rplm1[:1])) * vv ##:


# In[45]:



# In[46]:

# probe
vve.subz(uu, uue).smplf.subs(st.rev_tuple(rplm1[:1]))


# In[47]:

ffz


# ### Konstrunktion einer möglichst kurzen Kurve von der z3-Achse in den Ursprung
# 
# Drei Phasen:
# 1. $z_1$ vergrößern → $r_1$
# 2. mit $v_2=$const nach unten/oben "schrauben" ($z_2$ und $z_3$ verändern, bis $z_3=0$)
# 3. $z_1$ verkleinern
# 
# 
# \begin{align}
# %\label{eq_}
# L &= r_1 + \int_0^T \sqrt{\dot x_1^2 +  \dot x_2^2 + \dot x_3^2 }\, dt + r_1\\
# &= r_1 + \int_0^T \sqrt{(r_1 \dot z_2)^2 + \dot z_3^2 }\, dt + r_1\\
# &= 2 r_1 + \int_0^T \sqrt{  (r_1 v_2)^2 + r_1^4 v_2^2}\, dt \\
# &= 2 r_1 + (r_1 v_2)\sqrt{( 1+ r_1^2)} \int_0^T \, dt \\
# &= 2 r_1 + (r_1 v_2)\sqrt{( 1+ r_1^2)}T \\
# \end{align}
# 
# 
# $
# \Delta z_3 = - T r_1^2 v_2 \quad \Rightarrow \quad T = - \frac{\Delta  z_3 }{r_1^2 v_2}
# $
# 
# <br>
# 
# $
# \Rightarrow L= 2 r_1 - (r_1 v_2)\sqrt{( 1+ r_1^2)} \cdot \frac{\Delta  z_3 }{r_1^2 v_2}
# $

# In[48]:


r1, dz3 = sp.symbols('r1, \Delta{}z_3', positive=True) ##


# In[49]:

Le = sp.expand(2*r1-(r1*v2)*sp.sqrt(1+r1**2)*dz3/r1**2/v2) ##:
print Le
print Le.diff(r1)
# In[50]:

sol = sp.solve(Le.diff(r1), r1)


# In[51]:

# sp.plot(Le.subs(dz3, 2), (r1, 0, 5), ylim=(-1, 10))


# In[52]:

for i, s in enumerate(sol):
    print(i, s.subs(dz3, 1).evalf())


# In[53]:
def r1_opt(dz3_value):
    results = [s.subs(dz3, dz3_value).evalf() for s in sol]

    results.sort(key=lambda x: abs(sp.im(x)))

    r1 = sp.re(results[0])
    if r1 < 0:
        r1 = sp.re(results[1])
    assert r1 >= 0
    return r1

for p in [0.01, .1, 1, 10]:
    print(r1_opt(p))


# In[54]:

# Auswertung beschleunigen, durch Interpolation
zz3 = np.logspace(-3, 3, 100)

rr_opt = [r1_opt(z3_value) for z3_value in zz3]


# In[55]:

r1_opt_interp = sc.interpolate.interp1d(zz3, rr_opt, bounds_error=False, fill_value="extrapolate")


# #### optimaler Wert von $r_1$ in Abhängigkeit von $\Delta z_3$

# In[56]:

#plt.plot(zz3, r1_opt_interp(zz3))


# In[57]:

# Stickproben-Vergleich:

r1_opt_interp(.7) ##:
r1_opt(.7)##:


# ### Simulationsuntersuchung
# 
# #### Vektorfeld und Regelgesetz

# In[58]:

def rhs(state, _):
    u1, u2 = controller(state)
    x1, x2, x3 = state
    return np.array([u1, u2, x2*u1 - x1*u2])

vv_to_uu = sp.lambdify((v1, v2, x1, x2), list(uue))
z3_tol = 1e-2

def controller(state):
    x1, x2, x3 = state
    z1 = np.sqrt(x1**2 + x2**2)
    z3 = x3
    print ('x:{}'.format(x1,x2,x3))
    print('z:{}'.format(z1,z2,z3))

    r1_opt_value = r1_opt_interp(abs(z3))
    if z1 == 0:
        return [1, 0]

    if z1 < r1_opt_value and abs(z3) >= z3_tol:
        # Phase 1
        v1 = 1
        v2 = 0
    
    elif z1 >= r1_opt_value and abs(z3) >= z3_tol:
        # Phase 2
        v1 = 0
        v2 = np.sign(z3)
    elif abs(z3) < z3_tol:
        # Phase 3
        v1 = -1
        v2 = 0
    else:
        raise ValueError("Unexpected state: %s" % state)
        
    return vv_to_uu(v1, v2, x1, x2)


# In[31]:

def euler(rhs, y0, T, dt=.01):
    res=[y0]
    tt=[0]
    while tt[-1] <= T:
        x_old = res[-1]
        res.append(x_old+dt*rhs(x_old, 0))
        tt.append(tt[-1] + dt)
    return tt, np.array(res)
    
    


Asy = True
T = 4 # end time

tt = np.linspace(0, T, T*100+2)
xx0 = np.array([0, 0, 5])


controller(xx0)
rhs(xx0, 0)

def simulate(xx0,T):
    tt, xxn = euler(rhs, xx0, T, dt=0.01)# 0.01
    return np.array(xxn)



import matplotlib
from matplotlib import rc
matplotlib.rc('xtick', labelsize=44)
matplotlib.rc('ytick', labelsize=44)
matplotlib.rcParams.update({'font.size': 44})
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if Asy == True:

    N = 20
    np.random.seed(1)
    xx0_values = (np.random.rand(N, 3) - .5)*1

    res = [simulate(xx0,T) for xx0 in xx0_values]


    fig_1 = plt.figure(1)
    ax = fig_1.gca(projection='3d')
    fig_2 = plt.figure(2)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)

    fig_3 = plt.figure(3)
    ax4 = plt.subplot(221)
    ax5 = plt.subplot(222)
    ax6 = plt.subplot(223)

    for xxn in res:
        z1 = np.sqrt(xxn[:, 0] ** 2 + xxn[:, 1] ** 2)# r
        z2 = np.arctan2(xxn[:, 0], xxn[:, 1]) # phi
        z3 = xxn[:, 2] # z

        ax.plot(xxn[:, 0], xxn[:, 1], xxn[:, 2])
        ax.set_xlabel(r'$x_{1}$', fontsize=46, color='blue', labelpad=45)
        ax.set_ylabel(r'$x_{2}$', fontsize=46, color='blue', labelpad=45)
        ax.set_zlabel(r'$x_{3}$', fontsize=46, color='blue', labelpad=20)
        ax.grid(True)

        plt.sca(ax1)
        plt.plot(tt, xxn[:, 0], linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$x_{1}$', fontsize=46, labelpad=20)
        plt.sca(ax2)
        plt.plot(tt, xxn[:, 1], linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$x_{2}$', fontsize=46, labelpad=20)
        plt.sca(ax3)
        plt.plot(tt, xxn[:, 2], linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$x_{3}$', fontsize=46, labelpad=20)

        plt.sca(ax4)
        plt.plot(tt, z1, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$z_{1}$', fontsize=46, labelpad=20)
        plt.sca(ax5)
        plt.plot(tt, z2, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$z_{2}$', fontsize=46, labelpad=20)
        plt.sca(ax6)
        plt.plot(tt, z3, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$z_{3}$', fontsize=46, labelpad=20)
    plt.show()
    #
    # ax4.set_ylabel("$r$")
    # ax5.set_ylabel(r"$\varphi$")

if Asy ==False:
    xx0_values = np.array([0, 0, 5])
    r0 = xx0_values[0]
    phi0 = xx0_values[1]
    z0 = xx0_values[2]
    # r0 = xx0_values[0]
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    fig = plt.figure(2)
    res = simulate(xx0_values,T)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)

    # print tt,res
    x1 = res[:, 0]
    x2 = res[:, 1]
    x3 = res[:, 2]
    plt.sca(ax1)
    plt.plot(tt, x1, linewidth=3.0)
    plt.grid(True)
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$x_{1}$', fontsize=46)
    plt.sca(ax2)
    plt.plot(tt, x2, linewidth=3.0)
    plt.grid(True)
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$x_{2}$', fontsize=46)
    plt.sca(ax3)
    plt.plot(tt, x3, linewidth=3.0)
    plt.grid(True)
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$x_{3}$', fontsize=46)
    plt.ylim(-0.2, 5.2)
    # fig = plt.figure(3)
    # ax4 = plt.subplot(1,2,1)
    # ax5 = plt.subplot(1,2,2)
    # plt.sca(ax4)
    # plt.plot(range(len(U_List)), u, linewidth=3.0)
    # plt.grid()
    # plt.xlabel(r'$t(s)$', fontsize=46)
    # plt.ylabel(r'$u_{z}$', fontsize=46)
    # plt.sca(ax5)
    # plt.plot(range(len(U_List)), v, linewidth=3.0)
    # plt.grid()
    # plt.xlabel(r'$t(s)$', fontsize=46)
    # plt.ylabel(r'$v_{z}$', fontsize=46)

    plt.sca(ax)
    plt.plot(x1, x2, x3, linewidth=3.0)
    plt.grid()
    ax.set_xlabel(r'$x_{1}$', fontsize=46, color='blue', labelpad=45)
    ax.set_ylabel(r'$x_{2}$', fontsize=46, color='blue', labelpad=45)
    ax.set_zlabel(r'$x_{3}$', fontsize=46, color='blue', labelpad=20)

    if 1:
        #i = tt.tolist().index(1.25)
        #x11 = x1[i]
        #x21 = x2[i]
        #x31 = x3[i]
        #print 'r1,phi1,z1:{},{},{}'.format(r1, phi1, z1)
        x1end = x1[-1]
        x2end = x2[-1]
        x3end = x3[-1]
        print x1end,x2end,x3end
        ax.text(0,0,5.2, r'$t=0,x=(0,0,5)$', fontsize=40, color='green')
        ax.text(0.4, 0, 4.5, r'$Phase-I$', fontsize=30, color='green')
        ax.text(-1.3, 0, 4.5, r'$Phase-II$', fontsize=30, color='green')
        ax.text(x1end-0.5, x2end, x3end, r'$t=T_{end},x=(0,0,0)$', fontsize=40, color='green')
        ax.text(x1end-1, x2end, x3end-0.7, r'$Phase-III$', fontsize=30, color='green')
    # plt.xlabel(r'$r$', fontsize=46)
    # plt.ylabel(r'$\phi$', fontsize=46)
    # plt.zlabel(r'$z$', fontsize=46)
    plt.show()



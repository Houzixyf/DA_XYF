# coding: utf-8

# Ein Regelgesetz aus Literatur [1] für Brockett Integrator
# [1]: Liberzon, D.: Switching in systems and control. Springer Science & Business
# Media, 2012.

import time

time.ctime()

import sympy as sp
import numpy as np
import scipy as sc
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# get_ipython().magic(u'matplotlib inline')

import symbtools as st
from sympy.interactive import printing
from mpl_toolkits.mplot3d import Axes3D
printing.init_printing()

# In[37]:
def rhs(state,t,T_phase,r0):

    r, phi, z = state
    z_tol = 1e-2
    print state
        # if (t+1/r0<=0 and abs(t+1/r0)<1e-1):
        #     r = -1e1
        # #elif (t+1/r0>=0 and abs(t+1/r0)<1e-2):
        #     #r = 1e2
        # state[0] = r
    u, v = controller(state,z_tol,t,T_phase,r0)
        # print (state,t, r,u)
    if r == 0:
        return np.array([u, 0, r*v])
    else:
        a = np.array([u, v/r, r*v])
       #  print ('a:{}'.format(a))
        return a



def controller(state,z_tol,t,T_phase,r0):
    r, phi, z = state


    if t < T_phase and r0 < z_tol:
        u = 1#z0
        v = 0#z0


    else:
        # Phase 1
        u = -r**2
        v = -z

    return u,v



# In[31]:


# #### Euler-Vorwärts-Verfahren für Integration

# In[62]:

Asy = False
tt = np.linspace(0, 10, 101)
T_phase = 0.2
#controller(xx0)
#rhs(xx0, 0)
import matplotlib
from matplotlib import rc
matplotlib.rc('xtick', labelsize=44)
matplotlib.rc('ytick', labelsize=44)
matplotlib.rcParams.update({'font.size': 44})
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# In[63]:
if Asy == True:
    N = 20
    np.random.seed(1)
    xx0_values = 1*(np.random.rand(N, 3) - .5) * 1
    pose = np.where(xx0_values[:,0]<0)
    for i in pose:
        xx0_values[:,0][i]=abs(xx0_values[:,0][i])

    # r0 = xx0_values[0]
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    fig = plt.figure(2)
    res = []
    for xx0 in xx0_values:
        r0 = xx0[0]
        res_one = odeint(rhs, xx0, tt, args=(T_phase, r0))
        res.append(res_one)
    # res = [odeint(rhs, xx0, tt, args=(T_phase, r0)) for xx0 in xx0_values]
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    # print tt,res
    for xxn in res:
        r = xxn[:,0]
        phi = xxn[:,1]
        z = xxn[:,2]

        plt.sca(ax1)
        plt.plot(tt, r, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$r(m)$', fontsize=46)
        plt.sca(ax2)
        plt.plot(tt, phi, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$\varphi (rad)$', fontsize=46)
        plt.sca(ax3)
        plt.plot(tt, z, linewidth=3.0)
        plt.grid(True)
        plt.xlabel(r'$t(s)$', fontsize=46)
        plt.ylabel(r'$z(m)$', fontsize=46)

        plt.sca(ax)
        plt.plot(r, phi, z, linewidth=3.0)
        plt.grid()
        ax.set_xlabel(r'$r(m)$', fontsize=46, color='green', labelpad=30)
        ax.set_ylabel(r'$\varphi (rad)$', fontsize=46, color='green', labelpad=30)
        ax.set_zlabel(r'$z(m)$', fontsize=46, color='green', labelpad=30)
        ax.text2D(0.05, 0.95, "Dreidimensionale Abbildung der Trajektorie", transform=ax.transAxes)


        # In[64]:

    # Interaktives Plot-Fenster:
    # get_ipython().magic(u'matplotlib qt5')

    # Alternativ: Grafiken Eingebunden
    # %matplotlib inline


    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)
    #
    # fig = plt.figure(3)
    #
    #
    # for xxn in res:
    #     r = xxn[:,0]
    #     phi = xxn[:,1]
    #     z = xxn[:,2]
    #     ax1.plot(r)
    #     ax2.plot(phi)
    #     ax3.plot(z)

    plt.show()

if Asy == False:



    xx0_values = np.array([0,5,5])
    r0 = xx0_values[0]
    phi0 = xx0_values[1]
    z0 = xx0_values[2]
    # r0 = xx0_values[0]
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    fig = plt.figure(2)
    res = [odeint(rhs, xx0_values, tt, args=(T_phase,r0))]
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)


    # print tt,res
    r = res[0][:,0]
    phi = res[0][:,1]
    z = res[0][:,2]
    plt.sca(ax1)
    plt.plot(tt, r, linewidth=3.0)
    plt.grid()
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$r(m)$', fontsize=46)
    plt.sca(ax2)
    plt.plot(tt, phi, linewidth=3.0)
    plt.grid()
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$\varphi (rad)$', fontsize=46)
    plt.sca(ax3)
    plt.plot(tt, z, linewidth=3.0)
    plt.grid()
    plt.xlabel(r'$t(s)$', fontsize=46)
    plt.ylabel(r'$z(m)$', fontsize=46)
    plt.ylim((-0.2, 5.2))
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
    plt.plot(r, phi, z, linewidth=3.0)
    plt.grid()
    ax.set_xlabel(r'$r$', fontsize=46, color='blue', labelpad=45)
    ax.set_ylabel(r'$\varphi$', fontsize=46, color='blue', labelpad=45)
    ax.set_zlabel(r'$z$', fontsize=46, color='blue', labelpad=20)

    i = tt.tolist().index(T_phase)
    r1 = r[i]
    phi1 = phi[i]
    z1 = phi[i]
    print 'r1,phi1,z1:{},{},{}'.format(r1,phi1,z1)
    rend = r[-1]
    phiend = phi[-1]
    zend = z[-1]
    ax.text(0, 5, 5+0.1, r'$t=0~(r=0,\varphi=5,z=5)$', fontsize=40, color='green')
    ax.text(r1-0.11, phi1-150, z1, r'$t=T_{phase1}~(r=0.2,\varphi=5,z=5)$', fontsize=40, color='green')
    ax.text(rend-0.015, phiend-50, zend, r'$t=T_{end}~(r=0,\varphi=-241,z=0)$', fontsize=40, color='green')
    ax.text(0.15, 10, 5.2, r'$Phase-I$', fontsize=30, color='blue')
    ax.text(0.12, 0, 2.5, r'$Phase-II$', fontsize=30, color='blue')
    # plt.xlabel(r'$r$', fontsize=46)
    # plt.ylabel(r'$\phi$', fontsize=46)
    # plt.zlabel(r'$z$', fontsize=46)
    plt.show()




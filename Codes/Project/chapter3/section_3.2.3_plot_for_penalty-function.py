# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:18:54 2017

Abbildung für die Straf-Funktion Pe

@author: rst
"""

import math
import numpy as np
xmin = 0
xmax = 10
sk = np.linspace(-2,12,200)
m = 5
xmid = xmin + (xmax - xmin)/2
res = (sk-xmid)**2/(1 + 2.8**(m*(sk - xmin))) + (sk-xmid)**2/(1 +2.8**(m*(xmax - sk)))
y = 1*res

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rc('xtick', labelsize=58)
matplotlib.rc('ytick', labelsize=58)
matplotlib.rcParams.update({'font.size': 58})
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(sk,y, linewidth=3.0,label=r'$Pe$')
plt.plot(sk,(sk-xmid)**2,linewidth=3.0,label=r'$(k-kmid)^{2}$')
plt.xlabel(r'$k$', fontsize=58)
#plt.ylabel(r'$Pe$', fontsize=46)
plt.legend(loc='best', fontsize = 58)
plt.grid()

plt.subplot(1,2,2)
plt.semilogy(sk,y, linewidth=3.0,label=r'$Pe$')
plt.semilogy(sk,(sk-xmid)**2,linewidth=3.0,label=r'$(k-kmid)^{2}$')
plt.legend(loc='center', fontsize = 40)#lower right
plt.xlabel(r'$k$', fontsize=58)
plt.ylabel(r'$lg$', fontsize=58)
plt.grid()
plt.show()


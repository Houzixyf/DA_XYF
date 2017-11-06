# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:54:49 2017

@author: xyf
"""

import numpy as np
from math import e

kappa_list = np.linspace(0.1,12,20)
k_plus = 5
k_minus = 0
m = 4 / float(k_plus - k_minus)
psi = [k_plus - (k_plus-k_minus)/(1+e**(m*kappa))for kappa in kappa_list]

# from matplotlib import pyplot as plt
# plt.plot(kappa_list,psi)
# plt.ylim(ymax=5.5)


import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
from matplotlib import rc

    #a = sns.color_palette("husl", n_colors=14)
    #sns.set_palette(a)
matplotlib.rc('xtick', labelsize=58)
matplotlib.rc('ytick', labelsize=58)
matplotlib.rcParams.update({'font.size': 58})
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12  # default = 3.5, distance to major tick label in points

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.figure(1)
plt.grid()
plt.plot(kappa_list,psi,linewidth=3.0)
plt.xlabel(r'$\kappa$', fontsize=58)
plt.ylabel(r'$\psi$', fontsize=58)
plt.ylim(ymax=5.5)
plt.show()
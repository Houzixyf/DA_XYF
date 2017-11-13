# -*- coding: utf-8 -*-
"""
Abbildung für die Beschränkungs-Funktion k=psi
"""

import numpy as np
from math import e

kappa_list = np.linspace(-20,20,50)
k_plus = 6
k_minus = -6
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
plt.show()
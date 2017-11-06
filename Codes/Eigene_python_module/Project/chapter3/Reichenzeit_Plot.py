# -*- coding: utf-8 -*-

def r(t,r0):
    r = 1/(t+1/r0)
    print r


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
matplotlib.rc('xtick', labelsize=44)
matplotlib.rc('ytick', labelsize=44)
matplotlib.rcParams.update({'font.size': 44})
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12 # default = 3.5, distance to major tick label in points
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

data_1 = [0.92,0.08]
labels_1 = [r'$SA:\leq 16$',r'$SA:>16$']

data_2 = [0.0,1.0]
labels_2 = [r'$SA:\leq 64$',r'$SA:>64$']

# Make a normed histogram. It'll be multiplied by 100 later.
plt.figure()
plt.subplot(1, 2, 1)
colors = plt.cm.BuPu(np.linspace(0, 0.5, 2))
plt.bar(range(2), data_1, color=colors, tick_label=labels_1, align="center")

plt.legend()

formatter = FuncFormatter(to_percent)

# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
plt.grid()

plt.subplot(1, 2, 2)
colors = plt.cm.BuPu(np.linspace(0, 0.5, 2))
plt.bar(range(2), data_2, color=colors, tick_label=labels_2, align="center")

plt.legend()

formatter = FuncFormatter(to_percent)

# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
plt.grid()




# plt.bar(range(len(data)), data, tick_label=labels)
plt.show()















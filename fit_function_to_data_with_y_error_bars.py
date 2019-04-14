# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit function to data with y error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define a function
def func(x, a, b, c):
    return a * x**2 + b*x + c

# fabricate data with function + random noise
x = np.arange(0, 20, 1)
y = func(x,-1,7,15) + 0.8*x*np.random.randn(20)
dy = x+1    # y error bars increase with x

# fit data with function
popt, pcov = curve_fit(func, x, y, p0=None, sigma=dy)
popt

# create figure
fig, ax = plt.subplots(figsize=(8, 4))
plt.errorbar(x, y, xerr=0.1, yerr=dy, fmt='none', label='experiment')
plt.plot(x,func(x, *popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('displacement vs time')
ax.set_xlabel('time (ms)')
ax.set_ylabel('displacment (mm)')

plt.show()

print (popt)
print (pcov)

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit_function_to_data_with_y_error_bars.py
this script fits a defined function to a given data with y error bars
"""


import numpy as np
#from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
# read data from csv file
data = read_csv('sample01.csv', skiprows=1, header=None, sep=',', lineterminator='\n', names=["x","dx","y","dy"])
x = data.iloc[:,0]
dx = data.iloc[:,1]
y = data.iloc[:,2]
dy = data.iloc[:,3]
"""

# define a function
def linfunc(x, a, b):
    # a polynumial function
    return a*x + b

# define a function
def volume(r):
    # a polynumial function
    return 4/3*np.pi()*r**3

# fabricate data with function + random noise
# use in case there's no csv file ready
r = np.arange(1.0, 20.0, 1.0)
v = volume(r) * (1 + np.random.normal(0, 0.05, None))
dr = np.full((20), 0.2)
dv = 9/4*r**2


# fit data with function
popt, pcov = curve_fit(volume, r, v, p0=None, sigma=dv)
popt

# create figure
fig, ax = plt.subplots(figsize=(16, 8))
plt.errorbar(r, v, xerr=dr, yerr=dv, fmt='none', label='experiment')
plt.plot(r,volume(r, *popt), label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f' % tuple(popt))

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('displacement vs time')
ax.set_xlabel('time (ms)')
ax.set_ylabel('displacment (mm)')

plt.show()

#print (popt)
#print (pcov)

# printing the fit parameters with their error estimates
for i in range(0,3):
    print ('a' + str(i) + '= ' + str(popt[i]) + ' +/- ' + str(pcov[i,i]))

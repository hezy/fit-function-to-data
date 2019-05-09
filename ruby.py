# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit_function_to_data_with_y_error_bars.py
this script fits a defined function to a given data with y error bars
"""


import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
# read data from csv file
data = read_csv('ruby01.csv', skiprows=1, header=None, sep=',', lineterminator='\n', names=["x","dx","y","dy"])
x = data.iloc[:,0]
y = data.iloc[:,1]
"""
x = np.arange(670.0 ,700.0 ,0.01)


#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def doublet(x,a1,x1,s1,a2,x2,s2):
    return gauss_function(x, a1, x1, s1) + gauss_function(x, a2, x2, s2) 

# fabricate data
y = doublet(x, 100.0, 679.0, 0.2, 200.0, 680.0, 0.2) + np.random.normal(loc=1.0, scale=3.0, size=x.size)

# fit data with function
popt, pcov = curve_fit(doublet, x, y, p0=(100.0, 679.0, 0.2, 200.0, 680.0, 0.2), sigma=1*x/x)
popt


# configuring the figure
plt.close('all')
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(8, 12))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.3)
fst = 16 #font size for title
fsl = 14 #font size for axes labels

fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(x, y, '.b')
plt.plot(x,doublet(x, *popt), '-r')

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('Ruby')
ax.set_xlabel('Intensity (arb.)')
ax.set_ylabel('wavelength (nm)')

plt.show()

print (popt)
print (pcov)

"""
# printing the fit parameters with their error estimates
for i in range(0,3):
    print ('a' + str(i) + '= ' + str(popt[i]) + ' +/- ' + str(pcov[i,i]))
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
ref:
https://en.m.wikipedia.org/wiki/Voigt_profile
https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
http://journals.iucr.org/j/issues/2000/06/00/nt0146/index.html
"""

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

"""
# read data from csv file
data = read_csv('ruby01.csv', skiprows=1, header=None, sep=',', lineterminator='\n', names=["x","dx","y","dy"])
x = data.iloc[:,0]
y = data.iloc[:,1]
"""
x = np.arange(670.0 ,700.0 ,0.01)


# Normalized Gaussian
# x = 
def gauss_func(x, x0, sigma):
    return 1 / sigma / np.sqrt(2.0 * np.pi) * np.exp(-(x-x0)**2/(2.0 * sigma**2))

# Normalized Lorentzian
def loretz_func(x, x0, gamma):
    return gamma / np.pi / ((x-x0)**2 + gamma**2)

# Normalized Pseudo-Voigt 
# x0 = center position of the peak, eta = 
def pseudo_voigt_func(x , x0, eta, fwhm):   
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    gamma = fwhm / 2.0 
    return eta * loretz_func(x, x0 , gamma) + (1-eta) * gauss_func(x, x0, sigma)

# Normalized Voigt 
def voigt_func(x, x0, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    gamma = fwhm / 2.0 
    return np.real(wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
 

# fabricate data
y = 100.0 * voigt_func(x, 689.0, 2.0) + np.random.normal(loc=1.0, scale=3.0, size=x.size)

def fit_func(x, x0, a, e, f):
    return a * pseudo_voigt_func(x, x0, e, f)

# fit data with function
popt, pcov = curve_fit(fit_func, x, y, p0=(689.0, 100, 0.5, 2.0), sigma=1*x/x)
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
plt.plot(x,fit_func(x, *popt), '-r')

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('Ruby')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('Intensity (arb.)')

plt.show()

print (popt)
print (pcov)

"""
# printing the fit parameters with their error estimates
for i in range(0,3):
    print ('a' + str(i) + '= ' + str(popt[i]) + ' +/- ' + str(pcov[i,i]))
"""
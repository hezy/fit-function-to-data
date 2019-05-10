# -*- coding: utf-8 -*-
"""
Created on Th May 8, 2019
@author: Hezy Amiel
ref:
https://en.m.wikipedia.org/wiki/Voigt_profile
https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
http://journals.iucr.org/j/issues/2000/06/00/nt0146/index.html
http://journals.iucr.org/j/issues/1997/04/00/gl0484/gl0484.pdf
"""

import numpy as np
#from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

"""
# read data from csv file
data = read_csv('ruby01.csv', skiprows=1, header=None, sep=',', lineterminator='\n', names=["x","dx","y","dy"])
x = data.iloc[:,0]
y = data.iloc[:,1]
"""
x = np.arange(670.0 ,700.0 ,0.1)


# Voigt
# This function is the best description for a peak in spectroscopy and diffraction experiments.
# It is derived from the convolution of a Lorentzian function and a Gaussian function
def voigt(x, x0, w):
    sigma = w / np.sqrt(2.0 * np.log(2))
    gamma = w
    return np.real(wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)

# Gaussian
def gauss(x, x0, w):
    return np.exp(-1.0 * np.log(2.0) * ((x-x0)/2)**2)

# Lorentzian
def loretz(x, x0, w):
    return 1 / (1 + ((x-x0)/w)**2)

# Pseudo-Voigt
# We will try to aproximate the Voigt function with this simple linear combination of Gaussian and Lorentzian
# x0 = center position of the peak, eta = mixing parameter, 2w = full width at half maximum
def pseudo_voigt(x ,x0, I0, eta, w):
    return I0 * ( eta * loretz(x, x0 , w) + (1-eta) * gauss(x, x0, w) )


# fabricate data
y = 100.0 * voigt(x, 689.4, 2.0) + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x.size)
# dy = np.ones(x.size)


# fit data with function
popt, pcov = curve_fit(pseudo_voigt, x, y, p0=(691.0, 80.0, 0.5, 2.0), sigma=None)
popt


# configuring the figure
plt.close('all')
plt.rcParams.update({'font.size': 12})
fst = 16 #font size for title
fsl = 14 #font size for axes labels

fig, ax = plt.subplots(figsize=(14, 8))
plt.plot(x, y, '.b')
plt.plot(x,pseudo_voigt(x, *popt), '-r')

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
for i in range(x0, I0, eta, w):
    print ('a' + str(i) + '= ' + str(popt[i]) + ' +/- ' + str(pcov[i,i]))
"""

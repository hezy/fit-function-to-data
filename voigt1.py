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
x = np.arange(-10. ,10. ,0.05)


# Voigt
# This function is the best description for a peak in spectroscopy and diffraction experiments.
# It is derived from the convolution of a Lorentzian function and a Gaussian function
def voigt(x, x0, w):
    sigma = w / np.sqrt(2.0 * np.log(2.))
    gamma = w
    return np.real(wofz(((x-x0) + 1.0j*gamma)/sigma/np.sqrt(2.))) / sigma /np.sqrt(2.*np.pi)

# Gaussian
def gauss(x, x0, I0, w):
    sigma = w / np.sqrt(2.0 * np.log(2.))
    return I0 * (1. / sigma / np.sqrt(2.*np.pi)) * np.exp(-1.0 * ((x-x0)/(2.*sigma)**2.)

# Lorentzian
def lorentz(x, x0, I0, w):
    gamma = w
    return I0 / ((np.pi * gamma) * (1. + (np.square((x-x0)/gamma))))

# Pseudo-Voigt
# We will try to aproximate the Voigt function with this simple linear combination of Gaussian and Lorentzian
# x0 = center position of the peak, eta = mixing parameter, 2w = full width at half maximum
def pseudo_voigt(x ,x0, I0, eta, w):
    return I0 * ( eta * lorentz(x, x0 , w) + (1.0-eta) * gauss(x, x0, w) )


# fabricate data
y = 1. * voigt(x, 0.0, 2.0) # + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x.size)
# dy = np.ones(x.size)


# fit data with function


popt1, pcov1 = curve_fit(gauss, x, y, p0=(0.0, 1.0, 2.0), sigma=None)
popt2, pcov2 = curve_fit(lorentz, x, y, p0=(0.0, 1.0, 2.0), sigma=None)
popt3, pcov3 = curve_fit(pseudo_voigt, x, y, p0=(0.0, 1.0, 0.5, 2.0), sigma=None)

# configuring the figure
plt.close('all')
plt.rcParams.update({'font.size': 12})
fst = 16 #font size for title
fsl = 14 #font size for axes labels

fig, ax = plt.subplots(figsize=(14, 8))
plt.plot(x, y, '.')
plt.plot(x,gauss(x, *popt1), '-g')
plt.plot(x,lorentz(x, *popt2), '-b')
plt.plot(x,pseudo_voigt(x, *popt3), '-r')

# arange figure
ax.grid(True)
ax.legend(loc='best')
ax.set_title('Ruby')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('Intensity (arb.)')

plt.show()

print (popt1)
print (pcov1)

"""
# printing the fit parameters with their error estimates
for i in range(x0, I0, eta, w):
    print ('a' + str(i) + '= ' + str(popt[i]) + ' +/- ' + str(pcov[i,i]))
"""

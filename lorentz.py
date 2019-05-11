# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:13:38 2019
@author: hezya

https://en.m.wikipedia.org/wiki/Voigt_profile
http://journals.iucr.org/j/issues/1997/04/00/gl0484/gl0484.pdf
http://journals.iucr.org/j/issues/2000/06/00/nt0146/nt0146.pdf
"""

import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lorentz(x, w):
    # w = FWHM
    return 1. /(1. + np.square(x/w)) 

def gauss(x, w):
    return np.exp(-np.log(2.)*np.square(x/w))

def voigt(x, a, b, c):
    """
    for Lorentz a=b=28505100 c=np.sqrt(np.pi*np.log(2))*28505100
    for Gauss a=1, b=0, c=1
    """
    s = np.sqrt(np.log(2))
    return c * np.real(wofz(s*(a*x/2 + 1j*b)))

x = np.arange (-8.0, 8.0 , 0.1)
xfit = np.arange (-8.0, 8.0 , 0.01)

yL = lorentz(x,2.)
yG = gauss(x,2.)

guess = 28505100
popt_L, pcov_L = curve_fit(voigt, x, yL, p0=(guess, guess, guess*np.sqrt(np.pi*np.log(2))), sigma=None)
yVL = voigt(xfit, *popt_L)

popt_G, pcov_G = curve_fit(voigt, x, yG, p0=(1., 0, 1.), sigma=None)
yVG = voigt(xfit, 1., 0, 1.)

plt.close('all')
plt.grid(True)
plt.plot(x,yL,'or')
plt.plot(xfit,yVL,'-r')
plt.plot(x,yG,'ob')
plt.plot(xfit,yVG,'-b')
plt.show()

print(popt_L)
print(pcov_L)
print(popt_G)
print(pcov_G)

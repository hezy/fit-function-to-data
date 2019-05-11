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

def lorentz(x, gamma):
    return 1. / (1. + np.square(x/gamma)) 

def gauss(x, sigma):
    return np.exp(-np.log(2.)*np.square(x/sigma))

def voigt(x, a, b, c):
    return c * np.sqrt(np.log(2) * np.pi) * np.real(wofz(np.sqrt(np.log(2)) * (a*x + 1j*b)))

x = np.arange (-8.0, 8.0 , 0.1)
xfit = np.arange (-8.0, 8.0 , 0.01)

yL = lorentz(x,1.)
yG = gauss(x,1.)

popt_L, pcov_L = curve_fit(voigt, x, yL, p0=(1., 1., 1.), sigma=None)
yVL = voigt(xfit, *popt_L)

popt_G, pcov_G = curve_fit(voigt, x, yG, p0=(1., 1., 1.), sigma=None)
yVG = voigt(xfit, *popt_G)

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

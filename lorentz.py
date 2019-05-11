# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:13:38 2019
@author: hezya

https://en.m.wikipedia.org/wiki/Voigt_profile
http://journals.iucr.org/j/issues/1997/04/00/gl0484/gl0484.pdf
http://journals.iucr.org/j/issues/2000/06/00/nt0146/nt0146.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def lorentz(x, gamma):
    return 1. / (1. + np.square(x/gamma)) 

def gauss(x, sigma):
    return np.exp(-np.log(2.)*np.square(x/sigma))

def voigt(x, sigma, gamma):
    c =np.sqrt(np.log(2.))
    a = 2. * c * x / sigma 
    b = c * gamma / sigma
    return 2./sigma * np.sqrt(np.log(2.))/np.sqrt(np.pi) * wofz(a + 1j * b)


x = np.arange (-8.0, 8.0 , 0.01)

yL = lorentz(x,1.)
yG = gauss(x,1.)
yV = voigt(x,0.1,1.)

plt.close('all')
plt.grid(True)
plt.plot(x,yL,'.r')
plt.plot(x,yG,'.b')
plt.plot(x,yV,'.g')
plt.show()
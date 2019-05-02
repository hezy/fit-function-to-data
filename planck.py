# -*- coding: utf-8 -*-
"""
Created on Apr 30, 2019
@author: Hezy Amiel
planck.py
this script uses Planck's equation to create a set of black body radiation curves at varius temperatures, find the wavelength of maximum radiation for each curve, and use it to plot Wien's law
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
# Planck's constant
h = 6.62607015e-34 # J*s
 
# Boltzmann's constant
k = 1.380648528e-23 # J/K
 
# speed of light
c = 299792458 # m/s
 
 
# Palnck's function
# T = temperature, lam = wavelength (nm), l = wavelength (m)
def planck(T, lam):
    l = lam * 1e-9
    return (8*np.pi*h*c/l**5)/(np.exp(h*c/(l*k*T)-1))
 
# define a function
def wien(x, a):
    # a polynumial function
    return a/x

    
# create curves for different temperatures
x = np.arange(100, 10100, 1)
dx = x[1] - x[0]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(10, 14))
 
T = np.arange(500, 6000, 250)
Lmax = np.array([])
for t in T:
    y = planck(t,x)
    dydx = (np.gradient(y,dx))
    ddydx = (np.gradient(dydx,dx))
    ratio = abs(ddydx/dydx)
    axs[0].plot(x, y, "-", label=str(t))
    Lmax = np.append(Lmax,np.argmax(ratio))

# fit data with function
#popt, pcov = curve_fit(wien, x, y, p0=None, sigma=1)
#popt
   
# arange figure
 
axs[0].grid(True)
axs[0].set_title("Planck's curves")
axs[0].set_xlabel("wavelength (nm)")
axs[0].set_ylabel("intensity ()")
 
axs[1].plot(T, Lmax, 'o')
axs[1].grid(True)
axs[1].set_title("Wien’s Law")
axs[1].set_xlabel("Temperature (K)")
axs[1].set_ylabel("Lmax (nm)")
 
plt.show()
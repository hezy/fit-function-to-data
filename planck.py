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
def planck(lam, T):
    l = lam * 1e-9
    return (8*np.pi*h*c/l**5)/(np.exp(h*c/(l*k*T)-1))
 
    
# a function for fitting Wien's curve
def wien(x, a0, a1):
    return a0 + a1/x

    
# create Planck's curves for different temperatures
x = np.arange(100, 10100, 100)
xfit = np.arange(100, 10001, 1)
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(8, 12))
 
T = np.arange(1000, 7000, 500)
Lmax = np.array([])
for t in T:
    y = planck(x,t)
    axs[0].plot(x, y, ".")
    popt, pcov = curve_fit(planck, x, y, 2000)
    popt
    yfit = planck(xfit, *popt)
    axs[0].plot(xfit, yfit, "-")
    L = np.argmax(yfit)
    Lmax = np.append(Lmax,np.argmax(yfit))


# fit data with function
axs[1].plot(T, Lmax, "bo")
T2 = np.arange(1000, 7000, 100)
popt, pcov = curve_fit(wien, T, Lmax)
popt
axs[1].plot(T2, wien(T2, *popt), "r-")

# arange figure
 
axs[0].grid(True)
axs[0].set_title("Planck's curves")
axs[0].set_xlabel("wavelength (nm)")
axs[0].set_xlim(0,3000)
axs[0].set_ylabel("intensity ()")
 

axs[1].grid(True)
axs[1].set_title("Wien’s Law")
axs[1].set_xlabel("Temperature (K)")
axs[1].set_ylabel("Lmax (nm)")

plt.show()
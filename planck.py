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
    return 1e9*(8*np.pi*h*c/l**5)/(np.exp(h*c/(l*k*T)-1))
 
    
# a function for fitting Wien's curve
def wien(x, a0, a1):
    return a0 + a1/x

    
# create Planck's curves for different temperatures
Lam_min = 1 # nm
Lam_max = 30100 # nm
Lam_step = 100 # nm
x = np.arange(Lam_min, Lam_max , Lam_step)
xfit = np.arange(Lam_min, Lam_max, 1)
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(8, 12))

Tmin = 100 # K
Tmax = 10100 # K
Tstep = 1000 # K
T = np.arange(Tmin, Tmax, Tstep)
Lmax = np.array([])
for t in T:
    y = (planck(x,t))*(1 + np.random.normal(0, 0.1, None))
    axs[0].plot(x, y, ".")
    popt, pcov = curve_fit(planck, x, y, 2000)
    print(popt, pcov)
    yfit = planck(xfit, *popt)
    axs[0].plot(xfit, yfit, "-")
    L = np.argmax(yfit)
    Lmax = np.append(Lmax,np.argmax(yfit))


# fit 1/T to data 
axs[1].plot(T, Lmax, "bo")
T2 = np.arange(Tmin, Tmax, 100)
popt, pcov = curve_fit(wien, T, Lmax)
print(popt)
print(pcov)
axs[1].plot(T2, wien(T2, *popt), "r-")


# arange figure
 
axs[0].grid(True)
axs[0].set_title("Planck's curves")
axs[0].set_xlabel("wavelength (nm)")
axs[0].set_xlim(0,2000)
axs[0].set_ylabel("spectral radiance (W/sr/m^2/nm)")

axs[1].grid(True)
axs[1].set_title("Wien’s Law")
axs[1].set_xlabel("Temperature (K)")
axs[1].set_ylabel(r'$\lambda_{max} (nm)$')
axs[1].set_yscale('log')

plt.show()

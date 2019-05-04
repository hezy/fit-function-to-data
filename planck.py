# -*- coding: utf-8 -*-
"""
Created on Apr 30, 2019
@author: Hezy Amiel
planck.py
this script uses Planck's equation to create a set of black body radiation
curves at various temperatures, find λmax - the wavelength of maximum radiation 
for each curve, and use it to plot Wien's law: λmax = b/T
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
# Planck's constant
h = 6.62607015e-34 # J*s
 
# Boltzmann's constant
k = 1.380648528e-23 # J/K
 
# the speed of light in vacuum
c = 299792458 # m/s
 
"""
Palnck's function -
T = temperature, lam = wavelength (nm), l = wavelength (m)
this calculates the spectral radiance - the power per unit solid angle and 
per unit of area normal to the propagation, density of wavelength λ radiation 
per unit wavelegth at thermal equilibrium at temperature T.
"""
def planck(lam, T):
    l = lam * 1e-9
    return 1e-9*(8*np.pi*h*c/l**5)/(np.exp(h*c/(l*k*T)-1))
 
  
"""    
a function for fitting Wien's curve λmax = b/T
a is due to approximation error, b = Wien's constant (K*nm), to be evaluated. 
expected value is b = 2.8977729(17)e−3 K*m
"""
def wien(x, a, b):
    return a + b/x

# configuring the figure
plt.close('all')
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(8, 12))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.3)
fst = 16 #font size for title
fsl = 14 #font size for axes labels
    
# create Planck's curves for different temperatures
x = np.arange(1, 30101 , 100) # wavelength (nm)
xfit = np.arange(1, 30101, 1) # wavelength (nm)


T = np.arange(100, 10100, 1000) # temperature (K)
Lmax = np.array([])
for t in T:
    # creating Planck's cureve for temperature = t adding a random noise
    y = (planck(x,t))*(1 + np.random.normal(0, 0.1, None)) 
    axs[0].plot(x, y, ".") 
    
    # fitting Planck's curve for t
    popt, pcov = curve_fit(planck, x, y, 2000)
    print(popt, pcov)
    yfit = planck(xfit, *popt)
    axs[0].plot(xfit, yfit, "-")
    
    # finding the wavelength of maximal radiance for Planck's curve and appending to Wien's curve 
    L = np.argmax(yfit) 
    Lmax = np.append(Lmax,np.argmax(yfit))


# fitting Wien's curve with a 1/T function 
axs[1].plot(T, Lmax, "bo")
Tfit = np.arange(1, 12000, 100) # temperature (K)
popt, pcov = curve_fit(wien, T, Lmax)
print(popt)
print(pcov)
axs[1].plot(Tfit, wien(Tfit, *popt), "r-")


# aranging graphs in figure
 
axs[0].grid(True)
axs[0].set_title("Planck's curves", fontsize=fst)
axs[0].set_xlabel("wavelength (nm)", fontsize=fsl)
axs[0].set_xlim(0,2000)
axs[0].set_ylabel(r'$spectral \: radiance \: (W/sr/m^2/nm)$', fontsize=fsl)

axs[1].grid(True)
axs[1].set_title("Wien’s Law", fontsize=fst)
axs[1].set_xlabel("Temperature (K)", fontsize=fsl)
axs[1].set_xlim(1,10000)
axs[1].set_ylabel(r'$\lambda_{max} \: (nm)$',fontsize=fsl)
axs[1].set_yscale('log')

plt.show()
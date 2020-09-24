# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 23:28:40 2019
@author: hezya
"""

# importing libreries
import numpy as np
import matplotlib.pyplot as plt

# create tehoretical function and its noisy experimental counterpoint
T = np.arange(4.2,20,0.4)
R_theory = 1.23 + 0.5* T**2 
R_experiment = R_theory + np.random.normal(0, 2, np.size(T))

# prepare the graph
plt.close('all')
plt.figure(figsize=(11.7, 8.3), dpi=144)

plt.plot(T, R_experiment, 'o', label='experiment')
plt.plot(T, R_theory, '-', label='theory')

#plt.rc('text', usetex=False)
#plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
plt.grid(True)
plt.minorticks_on()
plt.xlabel('T (K)', fontsize=18)
plt.ylabel('R ($\Omega$)', fontsize=18)
plt.suptitle('Resistance vs. Temperature', y=0.97, fontsize=20)
plt.title('T = 20K $\\rightarrow$ 4.2K', y=1.0, fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')
#plt.set_xlim(0,20)
#plt.set_ylim(0,200)

plt.savefig('simple_graph')
plt.show()

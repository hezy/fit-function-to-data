# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 23:28:40 2019

@author: hezya
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,20,0.4)
y = x**2

fig_width_pt = 800                            # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27                     # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0            # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt        # width in inches
fig_height = fig_width*golden_mean            # height in inches
fig_size =  [fig_width,fig_height]

plt.close('all')
plt.figure(figsize=[fig_width,fig_height])

plt.plot(x, y, 'o', label='experiment')
plt.plot(x, y, '-', label='theory')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid(True)
plt.minorticks_on()
plt.xlabel('T (K)', fontsize=18)
plt.ylabel('R ($\Omega$)', fontsize=18)
plt.suptitle('$\mathrm{AuAgTe_4}$ - resistance vs. temperature', y=0.97, fontsize=20)
plt.title('P = 4.35GPa ,  T = 2K $\\rightarrow$ 300K', y=1.0, fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')
#plt.set_xlim(0,10)
#plt.set_ylim(0,100)

plt.savefig('graph')
plt.show()

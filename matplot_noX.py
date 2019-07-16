# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:18:10 2019

@author: hezya
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

x = np.arange(0.0, 12.1, 0.1)
y = x*x

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)
fig.savefig('temp.png')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:30:29 2022

@author: hezy
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

plt.style.use(['nature', 'notebook','grid'])

plt.plot(T, R_experiment, 'o', mfc='none', alpha=1, label='experiment')
plt.plot(T, R_theory, '-', alpha=0.5, label='theory')

plt.savefig('simple_graph')
plt.show()

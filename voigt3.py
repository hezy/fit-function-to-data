# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:53:03 2019

@author: hezya
"""

import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sympy import *

x = Symbol('x')
gamma = Symbol('gamma')
L = 1/(x**2 + gamma**2)
L_norm = Integral(L, (x, -oo, oo))

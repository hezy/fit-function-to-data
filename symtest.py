# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:18:38 2019
author: hezy1a
"""

from sympy import init_printing, symbols, Integral, integrate, oo

init_printing(use_unicode=True)

x, gamma = symbols ('x gamma')

def lorentz (x, gamma):
    return 1/(gamma**2 + x**2)

L_int = Integral(lorentz(x,gamma), (x, -oo, oo))
L_norm = integrate(lorentz(x,gamma), (x, -oo, oo))

display(L_int)
display(L_norm)
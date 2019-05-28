# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:18:38 2019
author: hezy1a
"""

from sympy import init_printing, symbols, Integral, oo, exp, Eq
init_printing(use_unicode=True)

x = symbols('x') 
gamma, sigma = symbols ('gamma sigma', positive=True)

def lorentz(x, gamma):
    return 1/(gamma**2 + x**2)

L_int = Integral(lorentz(x,gamma), (x, -oo, oo))
display(Eq(L_int, L_int.doit()))

def gauss(x,sigma):
    return exp(-x**2/2/sigma**2)

G_int = Integral(gauss(x,sigma), (x, -oo, oo))
display(Eq(G_int, G_int.doit()))


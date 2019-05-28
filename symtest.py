# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:18:38 2019
author: hezy1a
"""

from sympy import init_printing, symbols, Integral, oo, exp, Eq, solve, Rational, latex
from IPython.display import display
init_printing(use_unicode=True) #???

x = symbols('x') 
gamma, sigma = symbols ('gamma sigma', positive=True)

 
def lorentz(x, gamma):
    return 1/(gamma**2 + x**2)

# finding the norm of lorentz
L_int = Integral(lorentz(x,gamma), (x, -oo, oo))
L_norm = L_int.doit()
display(Eq(L_int, L_norm ))

# calculating the half width at half maximum (HWHM) for lorenz
L_HWHM = Eq(lorentz(x,gamma),Rational(1,2))
display(L_HWHM)
display(solve(L_HWHM, x))


def gauss(x,sigma):
    return exp(-x**2/2/sigma**2)

# finding the norm of gauss
G_int = Integral(gauss(x,sigma), (x, -oo, oo))
G_norm = G_int.doit()
display(Eq(G_int, G_norm))

# calculating the half width at half maximum (HWHM) for gauss
G_HWHM = Eq(gauss(x,sigma),Rational(1,2))
display(G_HWHM)
display(solve(G_HWHM, x))


# example: generating LaTeX code for the displayed equations:
#print(latex(Eq(L_int, L_norm )))
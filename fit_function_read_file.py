# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit_function_to_data_with_y_error_bars.py
this script fits a defined function to a given data with y error bars
"""


from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare


def func(x, a1, a2, a3):
    '''
    a polynumial function of x
    a1, a2, a3 are the coefficients
    '''
    return a1 * x**2 + a2*x + a3


def fab_data(x_min, x_max, x_step, rand_size):
    '''
    fabricate data with function + random noise
    use in case there's no csv file ready
    '''
    data = pd.DataFrame()
    data['x'] = np.arange(x_min, x_max, x_step)
    size = data.x.size
    a = 3 * np.random.randn(3)
    print('a = ' + str(a))
    data['dy'] = (data.x + 1) * np.random.randn(size)
    data['y'] = func(data.x, *a) + 0.76*rand_size * data.dy
    data['dx'] = np.full((size), 0.2)
    # y error bars increase with
    print(data)
    return data


def fit_it(func, data):
    '''
    fit data with function
    input: func, data
    returns a tuple: [optimal parameters, estimated parameters errors]
    '''
    popt, pcov = curve_fit(func, data.x, data.y, p0=None, sigma=data.dy)
    perr = np.sqrt(np.diag(pcov))
    chisq, p_val = chisquare(data.y, f_exp=func(data.x, *popt))
    return popt, perr, chisq, p_val


def plot_it(data, fit_param):
    '''
    input: data (Pandas DataFarme)
    output: a plot of the experimental results with the best fit
    '''
    fig, ax = plt.subplots(figsize=(14, 8), dpi=288)
    plt.plot(data.x, func(data.x, *fit_param[0]))
    # , label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f' % tuple(fit_param[0]))
    plt.errorbar(data.x, data.y, xerr=data.dx, yerr=data.dy)
    # arange figure
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_title('displacement vs time')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('displacment (mm)')
    return plt.show()


def round_to_error(x, Dx):
    '''
    This function rounds Dx to 2 significant digits, and rounds x to the same
    precision, and returns a string
    '''
    Dx_str = str('%s' % float('%.2g' % Dx))
    x_str = str(Decimal(str(x)).quantize(Decimal(Dx_str)))
    return x_str + ' +/- ' + Dx_str


def print_fit_results(fit_param):
    '''
    printing the fit parameters with their error estimates
    input: fit_param = [optimal parameters of fit, parameter estimated errors]
    returns:
    '''
    for i in range(0, 3):
        a = fit_param[0][i]
        Da = fit_param[1][i]
        print(f'a{i} = ' + round_to_error(a, Da))
    print('χ^2 = ' + str(fit_param[2]))
    print('p-value = ' + str(fit_param[3]))


# read data from csv file / fabricate new data
# data = pd.read_csv('sample01.csv', skiprows=0, header=0, sep=',')
DATA = fab_data(0, 20, 1, 1)
DATA.to_csv

# fit it
FIT_PARAM = fit_it(func, DATA)

# plot it
plot_it(DATA, FIT_PARAM)

# print fit results
print_fit_results(FIT_PARAM)
c
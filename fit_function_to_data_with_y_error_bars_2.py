# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit_function_to_data_with_y_error_bars.py
this script optimizes the free parametrs of a pre-defined function
to a given y vs x data with y error bars
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from decimal import Decimal


def func(x, a0, a1, a2):
    '''
    a polynumial function of x
    a0, a1, a2 are the coefficients
    '''
    return a0 + a1*x + a2 * x**2


def fab_data(x_min, x_max, x_step, rand_size):
    '''
    fabricate data with function + random noise
    use in case there's no csv file ready
    ''' 
    data = pd.DataFrame()
    data['x'] = np.arange(x_min, x_max, x_step)
    size = data.x.size
    data['dx'] = np.full((size), 0.2)
    a = 3 * np.random.randn(3)
    print('a = ' + str(a))
    data['dy'] = np.abs(0.05 * func(data.x, *a) * np.random.randn(size))
    data['y'] = func(data.x, *a) + rand_size * data.dy * np.random.randn(size )
    return data


def chi_2(observed_values, observed_errors, expected_values):
    test_statistic = 0
    for observed, errors, expected in zip(observed_values, observed_errors, expected_values):
        test_statistic += ((float(observed) - float(expected)) / float(errors))**2
    return test_statistic


def fit_it (func, data):
    '''
    fit data with function
    input: func, data
    returns a tuple: [optimal parameters, estimated parameters errors]    
    '''
    popt, pcov = curve_fit(func, data.x, data.y, p0=None, sigma=data.dy)
    perr = np.sqrt(np.diag(pcov))
    chi_square = chi_2(data.y, data.dy, func(data.x, *popt))
    degrees_freedom = data.y.size - popt.size
    chi_square_red = chi_square/degrees_freedom
    p_value = chi2.sf(chi_square, degrees_freedom)
    return popt, perr, chi_square, degrees_freedom, chi_square_red, p_value


def plot_it(data, fit_param, titles):
    '''
    input: data (Pandas DataFarme)
    output: a plot of the experimental results with the best fit 
    '''
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.errorbar(data.x, data.y, xerr=data.dx, yerr=data.dy,
                 fmt='none', label='experiment')
    plt.plot(data.x,func(data.x, *fit_param[0])) #, label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f' % tuple(fit_param[0]))
    # arange figure
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_title(titles[0])
    ax.set_xlabel(titles[1])
    ax.set_ylabel(titles[2])
    return plt.show()


def round_to_error(x, Dx):
    '''
    This function rounds Dx to 2 significant digits, and rounds x to the same
    precision, and returns a string
    '''
    Dx_str = str('%s' % float('%.2g' % Dx))
    x_str = str(Decimal(str(x)).quantize(Decimal(Dx_str)))
    return x_str + ' ± ' + Dx_str


def print_fit_results(data, fit_param):
    '''
    printing the fit parameters with their error estimates
    input: fit_param = [optimal parameters of fit, parameter estimated errors]
    returns:
    '''
    print(data)
    for i in range(0,3):
        a = fit_param[0][i]
        Da =  fit_param[1][i]
        print (f'a{i} = ' + round_to_error(a, Da)) 
    print('χ^2 = ' + round_to_error(fit_param[2], np.sqrt(2*fit_param[3])))
    print('degrees of freedom = ' + str(fit_param[3]))
    print('χ^2_red = ' + round_to_error(fit_param[4], np.sqrt(2/fit_param[3])))
    print('p-value = ' + str(fit_param[5])) 

        
''' read data from csv file / fabricate new data '''
DATA = pd.read_csv('sample02.csv', skiprows=0, header=0, sep=',')
# DATA = fab_data(0, 30, 1, 1)

# fit it
FIT_PARAM = fit_it(func,DATA)

# plot it
TITLES = 'Displacment vs Time', 'Time (ms)', 'Displacement (mm)' 
plot_it(DATA, FIT_PARAM, TITLES)

# print fit results
print_fit_results(DATA, FIT_PARAM)


    
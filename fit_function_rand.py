# -*- coding: utf-8 -*-
"""
Created on Sat Mar 9, 2019
@author: Hezy Amiel
fit_function_to_data_with_y_error_bars.py
this script fits a defined function to a given data with y error bars
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
#from scipy.stats import chisquare
from decimal import Decimal


def poly_2(x, a0, a1, a2):
    '''
    a polynumial function of x
    a0, a1, a2 are the coefficients
    '''
    return a0 + a1*x + a2 * x**2


def zero_func(x, a0, a1, a2):
    return poly_2(x, a0, a1, a2) - poly_2(x, a0, a1, a2)



def noise(signal, sigma_background, sigma_measurment):
    '''
    Adds noise to a given signal
    sigma_background is the standart deviation of a normal background noise
    sigam_measurment is the standart deviation of a normal noise proportional
    to the signal
    '''
    noisy_signal = (signal
                    + np.random.normal(0, sigma_background, np.size(signal))
                    + signal * np.random.normal(0,
                                                sigma_measurment,
                                                np.size(signal)))
    return noisy_signal


def fab_data(func, x_min, x_max, x_step, rand_size):
    '''
    fabricate data with function + random noise
    use in case there's no csv file ready
    '''
    data = pd.DataFrame()
    data['x'] = np.arange(x_min, x_max, x_step)
    size = data.x.size
    a = 3 * np.random.randn(3)
    print('a = ' + str(a))
    data['dy'] = (data.x + 1) * np.abs(np.random.randn(size))
    print(data.dy)
    data['y'] = func(data.x, *a) + 1 * data.dy * np.random.randn(size)
    data['dx'] = np.full((size), 0.2)
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
    chi2 = np.sum(((data.y - func(data.x, *popt))/data.dy)**2)
    dof = np.size(data.x) - np.size(popt)
    chi2_red = chi2 / dof
    P_value = 1 - stats.chi2.cdf(chi2, dof)
    return popt, perr, chi2, dof, chi2_red, P_value


def calc_residuals(func, data, fit_param):
    residuals = pd.DataFrame()
    residuals = data
    residuals.y = data.y - func(data.x, *fit_param)
    return residuals


def plot_it(data, func, fit_param, titles):
    '''
    input: data (Pandas DataFarme)
    output: a plot of the experimental results with the best fit
    '''
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.errorbar(data.x, data.y, xerr=data.dx, yerr=data.dy,
                 fmt='none', label='experiment')
    plt.plot(data.x, func(data.x, *fit_param[0]),
             label='fit')
    props = dict(boxstyle='round', facecolor='ivory', alpha=0.3)
    text_box = text_fit_results(fit_param)
    ax.text(0.05, 0.85, text_box, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
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
    return x_str + ' +/- ' + Dx_str


def text_fit_results(fit_param):
    '''
    creating a text string with the fit parameters with their error estimates
    and relevant statistical tests for goodnes of fit.
    input: fit_param = [optimal parameters of fit, parameter estimated errors]
    returns: a text with the fit results
    '''
    textstr = ['?', '?', '?', '?', '?', '?', '?']
    for i in range(0, 3):
        a = fit_param[0][i]
        Da = fit_param[1][i]
        textstr[i] = f'a{i} = ' + round_to_error(a, Da)
    textstr[3] = 'χ^2 = ' + str(fit_param[2])
    textstr[4] = 'dof = ' + str(fit_param[3])
    textstr[5] = 'χ^2_red = ' + str(fit_param[4])
    textstr[6] = 'P-value = ' + str(fit_param[5])
    text = '\n'.join(textstr)
    return text


# read data from csv file / fabricate new data
# data = pd.read_csv('sample01.csv', skiprows=0, header=0, sep=',')
DATA = fab_data(poly_2, 0, 20, 1, 1)

# fit it
FIT_PARAM = fit_it(poly_2, DATA)

# plot it
TITLES = 'Displacment vs Time', 'Time (ms)', 'Displacement (mm)'
plot_it(DATA, poly_2, FIT_PARAM, TITLES)

TITLES = 'Displacment residuals vs Time', 'Time (ms)', 'y_{obs} - y_{fit} (mm)'
plot_it(calc_residuals(poly_2, DATA, FIT_PARAM[0]), zero_func, FIT_PARAM, TITLES)

# print fit results
print(text_fit_results(FIT_PARAM))

# -*- coding: utf-8 -*-
"""
Created on 2019-06-19
@author: Hezy Amiel
"""

import glob
import re
import numpy as np
import pandas as pd
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def format_title(raw_title):
    '''
    input - a string for title/subtitle
    output - the formated string with nice TeX chemical forumals
    and scientific notation
    '''
    formated_title = raw_title.replace(r'SUBTITLE, ', '').replace(r'TITLE, ', '').replace(r'LEGEND, ', '').strip('\n')
    formated_title = re.sub(r'(.+)_(\d)',
                            r'$\\mathrm{\1_\2}$',
                            formated_title)
    formated_title = re.sub(r'([\-0-9.])+e([\-0-9.]+)',
                            r'$\1 \\times 10^{\2} $',
                            formated_title)
    formated_title = formated_title.replace('->', '$\\rightarrow$')
    return formated_title


def extended_title(file):
    '''
    input - file name
    output - tuple with title, subtitle, legend
    '''
    data_file = open(file, "r")
    my_list = []
    for line in data_file:
        my_list.append(line)
    data_file.close()
    return str(my_list[1]).split(';')


def fig_size(fig_width_pt, dpi):
    '''
    input - fig_width_pt (get it from from LaTeX using \showthe\columnwidth)
    output - fig_size
    '''
    inches_per_pt = 1.0/dpi                     # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0            # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt        # width in inches
    fig_height = fig_width * golden_mean            # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


plt.close('all')
plt.figure(figsize=fig_size(800, 72), dpi=72)

files = glob.glob('./*.txt')
for file in sorted(files):
    data = pd.read_csv(file,
                       skiprows=5,
                       header=1,
                       sep=',',
                       lineterminator='\n',
                       index_col=False)

    # y vs x plot:
    legend = format_title(extended_title(file)[2])
    x = data.iloc[:, 1]
    Dx = data.iloc[:, 2]
    y = data.iloc[:, 3]
    Dy = data.iloc[:, 4]
    plt.errorbar(x, y, xerr=Dx, yerr=Dy, fmt=".", capsize=4, label=legend)


plt.grid(True, which='major', axis='both',
         color='grey', linestyle='-', linewidth=0.5)
plt.grid(True, which='minor', axis='both',
         color='grey', linestyle='-', linewidth=0.25)
plt.minorticks_on()

plt.rc('text', usetex=False)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
plt.xlabel(data.iloc[:, 1].name, fontsize=18)
plt.ylabel(data.iloc[:, 3].name, fontsize=18)
title = format_title(extended_title(file)[0])
subtitle = format_title(extended_title(file)[1])

plt.suptitle(title, y=0.97, fontsize=20)
plt.title(subtitle, y=1.0, fontsize=18)
plt.xticks(np.arange(0, 12, 2), fontsize=16)
plt.yticks(fontsize=16)

plt.legend(loc='upper left')
#plt.xlim(0, 12)
#plt.ylim(0, 4)

plt.savefig('y_vs_x_plot.png')
plt.show()
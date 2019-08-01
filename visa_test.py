# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:31:34 2019

@author: Hezy
"""

import visa
rm = visa.ResourceManager()
print(rm.list_resources())
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:42:25 2024

@author: willy
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

x = np.arange(0,101,1)/100

radicale = np.sqrt(x*(1-x))
radicale[radicale==0] = np.sqrt(1e-7)

logaritmo = np.log(radicale)/np.log(np.sqrt(1e-7))

plt.figure()
plt.plot(x,logaritmo)
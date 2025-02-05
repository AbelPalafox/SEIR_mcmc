#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:42:08 2025

@author: abel
"""


import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import sys
from datetime import datetime
from SEIR_emcee import SEIR_emcee

args = sys.argv[1:]

input_data_file = args[0]
# reading data into a dataframe
df_input = read_csv(input_data_file)

print(df_input)

dates = df_input['Date']
data= df_input['Smoothed_Daily_Cases']

ini_date = datetime.strptime(dates[0],'%Y-%m-%d')
times = np.array([ (datetime.strptime(date_i,'%Y-%m-%d') - ini_date).days for date_i in dates ])

# setting initial values
N = 100000

E0 = 0 
I0 = data[0]
R0 = 0
S0 = N - E0 - I0 - R0

x0 = [S0,E0,I0,R0]

T = 100000

nwalkers = 6
ndim = 3

seir_emcee = SEIR_emcee(
    ndim = ndim,
    nwalkers = nwalkers,
    N=N,
    data=data,
    time=times,
    x0=x0,
    alpha_beta_prior=1.0,
    beta_beta_prior = 1.0,
    alpha_sigma_prior = 1.0,
    beta_sigma_prior = 1.0,
    alpha_gamma_prior = 1.0,
    beta_gamma_prior = 1.0,
    likelihood_model = 'Gaussian'
    )


theta_0 = 0.5+np.random.randn(nwalkers, ndim) * 1e-4

seir_emcee.run(theta_0, T)

fig,ax = plt.subplots(ndim,1)
res = [ax[i].plot(seir_emcee.sampler.chain[:,:,i].T, '-', color='k', alpha=0.3) for i in range(ndim)]
#res = [ax[i].axhline(theta_true[i]) for i in range(ndim)]


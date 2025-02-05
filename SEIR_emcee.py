#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:36:28 2025

@author: abel
"""
from tqdm import tqdm
import emcee
import corner
import numpy as np
from SEIR_mcmc_base import SEIR_mcmc_base

class SEIR_emcee(SEIR_mcmc_base) :
    
    def __init__(self, *argv, **kwargs) :
        
        super().__init__(self, *argv, **kwargs)
        
        self.nwalkers = kwargs['nwalkers']
        self.ndim = kwargs['ndim']
        print(self.nwalkers, self.ndim)
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
        
    def lnprob(self, theta) :
        
        
        if not self.Supp(theta) :
            #print('Out of support!')
            return -np.inf
        
        if self.likelihood_model == 'Poisson' :
            lnlike = self.LikelihoodEnergyPoisson(theta)
        else :
            lnlike = self.LikelihoodEnergyGaussian(theta)
        
        lnprior = self.PriorEnergy(theta)

        return lnlike + lnprior
    
    def run(self, theta_0, T) :
        #print(theta_0)
        
        with tqdm(total=T) as pbar:
            for i, _ in enumerate(self.sampler.sample(theta_0, iterations=T)):
                pbar.update(1)
        
        
        return True
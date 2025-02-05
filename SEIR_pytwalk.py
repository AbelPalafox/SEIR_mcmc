#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:33:27 2025

@author: abel
"""
from pytwalk import pytwalk
from SEIR_Model import SEIR_Model
from SEIR_mcmc_base import SEIR_mcmc_base
import scipy
import numpy as np

# defining the class for the SEIR_twalk model
class SEIR_pytwalk(pytwalk,SEIR_mcmc_base) :

    def __init__(self, *argv, **kwargs) :
        
        SEIR_mcmc_base.__init__(self, *argv, **kwargs)
        
        if self.likelihood_model == 'Poisson' :    
            super().__init__(argv[0],k=1,w=self.LikelihoodEnergyPoisson,Supp=self.Supp,u=self.PriorEnergy)
        else :
            super().__init__(argv[0],k=1,w=self.LikelihoodEnergyGaussian,Supp=self.Supp,u=self.PriorEnergy)


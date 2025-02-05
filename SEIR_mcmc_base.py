#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:12:53 2025

@author: abel
"""
from SEIR_Model import SEIR_Model
import scipy
import numpy as np

class SEIR_mcmc_base :
        
    def __init__(self, *argv, **kwargs) :
        
        try :
            self.N = kwargs['N']
            self.time = kwargs['time']
            self.x0  = kwargs['x0']
            self.data = kwargs['data']
            self.likelihood_model = kwargs['likelihood_model']
        except :
            print('Warning. Something is strange here!')
            
        self.alpha_beta_prior = kwargs['alpha_beta_prior']
        self.beta_beta_prior = kwargs['beta_beta_prior']
        self.alpha_sigma_prior = kwargs['alpha_sigma_prior']
        self.beta_sigma_prior = kwargs['beta_sigma_prior']
        self.alpha_gamma_prior = kwargs['alpha_gamma_prior']
        self.beta_gamma_prior = kwargs['beta_gamma_prior']
                    
    def LikelihoodEnergyPoisson(self, theta) :
        
        #print('evaluating likelihood poisson')
        
        N = self.N
        t = self.time
                
        beta, sigma, gamma = theta
        
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        x = seir_model.run(self.x0,t)
        
        S, E, I, R = x[:]
        
        incidency = seir_model.incidency(I,t,1)
        
        p_i = -incidency + self.data[1:]*np.log(np.abs(incidency))
        
        return np.sum(p_i)
    
    def LikelihoodEnergyGaussian(self,theta) :
        
        #print('evaluating likelihood gaussian')
        N = self.N
        t = self.time
                
        beta, sigma, gamma = theta
                
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        x = seir_model.run(self.x0,t)
        
        S, E, I, R = x[:]
        
        # assuming 
        val = np.linalg.norm(I-self.data)**2.0
           
        return val
    
    def PriorEnergy(self,theta) :
        #print('evaluating prior***')
        
        beta, sigma, gamma = theta
        
        alpha_beta_prior = self.alpha_beta_prior
        beta_beta_prior = self.beta_beta_prior
        alpha_sigma_prior = self.alpha_sigma_prior
        beta_sigma_prior = self.beta_sigma_prior
        alpha_gamma_prior = self.alpha_gamma_prior
        beta_gamma_prior = self.beta_gamma_prior
        
        log_pri_beta = scipy.stats.beta.logpdf(beta,alpha_beta_prior,beta_beta_prior)
        log_pri_sig = scipy.stats.beta.logpdf(sigma,alpha_sigma_prior,beta_sigma_prior)
        log_pri_gam = scipy.stats.beta.logpdf(gamma,alpha_gamma_prior,beta_gamma_prior)
        
        return (float(log_pri_beta+log_pri_sig+log_pri_gam))
        
    def Supp(self, theta) :
    
        if (theta <= 0).any() :
            return False
            
        if (theta > 1).any() :
            return False
        
        return True
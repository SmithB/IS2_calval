#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:06:47 2018

@author: ben
"""
import numpy as np
 

class waveform(object):
    __slots__=['p','t','t0','dt', 'tc']
    def __init__(self,p, t, t0=None, tc=None):
        if t0 is not None:
            if t0.size < self.p.size[1]:
                self.t0=np.zeros(self.p.size[1])+t0
            else:
                self.t0=t0
        if tc is not None:
            if tc.size < self.p.size[1]:
                self.tc=np.zeros(self.p.size[1])+tc
            else:
                self.tc=tc
        self.t=t
        self.dt=t[1]-t[0]
        self.p=p
        if p.ndim == 1:
            self.p.shape=[p.size,1]
        self.size=self.p.shape[1]
        self.nSamps=self.p.size[0]
    
    def  __getitem__(self, key):
        return waveform(self.p[:,key], self.t, t0=self.t0[key])
        
    def centroid(self, els=None):
        """
        Calculate the centroid of a distribution, optionally for the subset specified by "els"
        """
        if els is None:
            els=np.ones_like(self.p, dtype=bool)
        return np.sum(self.t[els]*self.p[els])/self.p[els].sum()

    def sigma(self, els=None, C=None):
        """
        Calculate the standard deviation of the energy in a distribution,  optionally for the subset specified by "els"
        """
        if els is None:
            els=np.ones_like(self.t, dtype=bool)
        if C is None:
            C=self.centroid(els)
        return np.sqrt(np.sum(((self.t[els]-C)**2)*self.p[els])/self.p[els].sum())
    
    def percentile(self, P, els=None):
        """
        Calculate the specified percentiles of a distribution,  optionally for the subset specified by "els"
        """
        if els is not None:
            C=np.cumsum(self.p[els])
            return np.interp(P, C/C[-1], self.t[els]) 
        else:
            C=np.cumsum(self.p)
            return np.interp(P, C/C[-1], self.t) 
    
    def robust_spread(self, els=None):
        """
        Calculate half the difference bewteen the 16th and 84th percentiles of a distribution
        """
        lowHigh=self.percentile(np.array([0.16, 0.84]), els=els)
        return (lowHigh[1]-lowHigh[0])/2.
        
    def nSigmaMean(self, N=3, els=None, tol=None, maxCount=20):
        """
            Calculate the iterative N-sigma edit, using the robust spread to measure sigma
        """
        if tol is None:
            tol=0.1*(self.t[1]-self.t[0])
        if els is None:
            els=self.p>0
        else:
            els = els & (self.p > 0)
        t_last=self.t[0]
        tc=self.centroid(els)  
        sigma=self.robust_spread( els)
        count=0
        while (np.abs(t_last-tc) > tol) and (count<maxCount):
            count+=1
            these=(self.p > 0) & (np.abs(self.t-tc) < N*sigma)
            t_last=tc;
            tc=self.centroid(els=these)
            sigma=self.robust_spread(els=these)
        return tc, sigma

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:06:47 2018

@author: ben
"""
import numpy as np
 

class waveform():
    def __init__(self,p, t, t0=None):
        self.p=p
        if t0 is not None:
            if t0.size < self.p.size[1]:
                self.t0=np.zeros(self.p.size[1])+t0
            else:
                self.t0=t0
        self.t=t
        self.size=self.p.size[1]
        self.nSamps=self.p.size[0]
        
def wf_centroid(WF, els=None):
    """
    Calculate the centroid of a distribution, optionally for the subset specified by "els"
    """
    if els is None:
        els=np.ones_like(WF['p'], dtype=bool)
    return np.sum(WF['t'][els]*WF['p'][els])/WF['p'][els].sum()

def wf_sigma(WF, els=None, C=None):
    """
    Calculate the standard deviation of the energy in a distribution,  optionally for the subset specified by "els"
    """
    if els is None:
        els=np.ones_like(WF['t'], dtype=bool)
    if C is None:
        C=wf_centroid(WF, els)
    return np.sqrt(np.sum(((WF['t'][els]-C)**2)*WF['p'][els])/WF['p'][els].sum())

def wf_percentile(WF, P, els=None):
    """
    Calculate the specified percentiles of a distribution,  optionally for the subset specified by "els"
    """
    if els is not None:
        C=np.cumsum(WF['p'][els])
        return np.interp(P, C/C[-1], WF['t'][els]) 
    else:
        C=np.cumsum(WF['p'])
        return np.interp(P, C/C[-1], WF['t']) 

def wf_robust_spread(WF, els=None):
    """
    Calculate half the difference bewteen the 16th and 84th percentiles of a distribution
    """
    lowHigh=wf_percentile(WF, np.array([0.16, 0.84]), els=els)
    return (lowHigh[1]-lowHigh[0])/2.
    
def nSigmaMean(WF, N=3, els=None, tol=None, maxCount=20):
    """
        Calculate the iterative N-sigma edit, using the robust spread to measure sigma
    """
    if tol is None:
        tol=0.1*(WF['t'][1]-WF['t'][0])
    if els is None:
        els=WF['p']>0
    else:
        els = els & (WF['p'] > 0)
    t_last=WF['t'][0]
    tc=wf_centroid(WF, els)  
    sigma=wf_robust_spread(WF, els)
    count=0
    while (np.abs(t_last-tc) > tol) and (count<maxCount):
        count+=1
        these=(WF['p'] > 0) & (np.abs(WF['t']-tc) < N*sigma)
        t_last=tc;
        tc=wf_centroid(WF, els=these)
        sigma=wf_robust_spread(WF, els=these)
    return tc, sigma

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:06:47 2018

@author: ben
"""
import numpy as np

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

class waveform(object):
    __slots__=['p','t','t0', 'dt', 'tc', 'size', 'nSamps', 'nPeaks','shots','params']
    def __init__(self, t, p, t0=0, tc=0, nPeaks=1, shots=np.NaN):

        self.t=t
        self.t.shape=[t.size,1]
        self.dt=t[1]-t[0]
        self.p=p
        if p.ndim == 1:
            self.p.shape=[p.size,1]
        self.size=self.p.shape[1]
        self.nSamps=self.p.shape[0]
        self.params=dict()

        kw_dict={'t0':t0, 'tc':tc, 'nPeaks':nPeaks,'shots':shots}
        for key, val in kw_dict.items():
            if ~hasattr(val,'__len__') or val.size < self.size:
                setattr(self, key, np.zeros(self.size, dtype=np.array(val).dtype)+val)
            else:
                setattr(self, key, val)

    def __getitem__(self, key):
        return waveform(self.t, self.p[:,key], t0=self.t0[key], tc=self.tc[key], nPeaks=self.nPeaks[key],shots=self.shots[key])

    def centroid(self, els=None, threshold=None):
        """
        Calculate the centroid of a distribution, optionally for the subset specified by "els"
        """
        if els is not None:
            return np.sum(self.t[els]*self.p[els])/self.p[els].sum()
        if threshold is not None:
            p=self.p.copy()
            p[p<threshold]=0
            p[~np.isfinite(p)]=0
            return np.sum(self.t*p, axis=0)/np.sum(self.p, axis=0)
        return  np.sum(self.t*self.p, axis=0)/np.sum(self.p, axis=0)

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

    def count_peaks(self, threshold=0.25, W=3, return_indices=False):
        K=gaussian(np.arange(-3*W, 3*W+1), 0, W)
        N=np.zeros(self.size)
        if return_indices:
            peak_list=list()
        for col in range(self.size):
            pS=np.convolve(self.p[:,col], K,'same')
            peaks=(pS[1:-1] > pS[0:-2]) & (pS[1:-1] > pS[2:]) & (pS[1:-1] > np.nanmax(pS)*threshold)
            N[col]=peaks.sum()
            if return_indices:
                peak_list.append(np.where(peaks)[0]+1)
        if return_indices:
            return N, peak_list
        else:
            return N


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
        sigma=self.robust_spread(els)
        count=0
        while (np.abs(t_last-tc) > tol) and (count<maxCount):
            count+=1
            these=(self.p > 0) & (np.abs(self.t-tc) < N*sigma)
            t_last=tc;
            tc=self.centroid(els=these)
            sigma=self.robust_spread(els=these)
        return tc, sigma

    def subBG(self, bg_samps=np.arange(0,30, dtype=int), t50_minus=None):
        """ subtract a background estimate from each trace

        For each individual waveform, calculate an estimate of the bacground estimate.
        Two options allowed are:
            -specify samples with bg_samps (default = first 30 samples of the trace)
            -specify t50_minus: samples earlier than the trace's t50() minus
                t50_minus are used in the background calculation
        """
        if t50_minus is not None:
            t50=self.t50()
            bgEst=np.zeros(self.size)
            for ii in range(self.size):
                bgind=np.where(self.t < t50[ii]-t50_minus)[0]
                if len(bgind) > 1:
                    bgEst[ii]=np.nanmean(self.p[bgind, ii])
        else:
            bgEst=np.nanmean(self.p[bg_samps, :], axis=0)
        self.p=self.p-bgEst
        return self

    def normalize(self):
        self.subBG()
        self.p=self.p/np.nanmax(self.p, axis=0)
        return self

    def t50(self):
        t50=np.zeros(self.size)
        for col in np.arange(self.size):
            p=self.p[:,col]
            p50=np.nanmax(p)/2
            i50=np.where(p>p50)[0][0]
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            t50[col] = self.t[i50-1] + dp*self.dt
        return t50

    def fwhm(self):
        FWHM=np.zeros(self.size)
        for col in np.arange(self.size):
            p=self.p[:,col]
            p50=np.nanmax(p)/2
            # find the elements that have power values greater than 50% of the max
            ii=np.where(p>p50)[0]
            i50=ii[0]
            # linear interpolation between the first p>50 value and the last p<50
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            temp = self.t[i50-1] + dp*self.dt
            # linear interpolation between the last p>50 value and the next value
            i50=ii[-1]+1
            dp=(p50 - p[i50-1]) / (p[i50] - p[i50-1])
            FWHM[col] = self.t[i50-1] + dp*self.dt - temp            
        return FWHM
    
    def calcMean(self, threshold=255):
        good=np.sum( (~np.isfinite(self.p)) & (self.p < threshold), axis=0) < 2
        return waveform(self.t, np.nanmean(self[good].normalize().p, axis=1))
       
      


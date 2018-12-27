#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:01:44 2018

@author: ben
"""

import numpy as np
import h5py
from waveform import waveform

def make_rx_scat_catalog(TX, h5_file=None):
    """
    make a dictionary of waveform templates by convolving the transmit pulse with 
    subsurface-scattering SRFs
    """
    if h5_file is None:
        h5_file='/Users/ben/Dropbox/ATM_red_green/subsurface_srf_no_BC.h5'
    with h5py.File(h5_file,'r') as h5f:
        t0=np.array(h5f['t'])*1.e9;
        z=np.zeros_like(t0)
        z[np.argmin(abs(t0))]=1;
        TXc=np.convolve(TX.p.ravel(), z, 'full')    
        t_full=np.arange(TXc.size)*0.25
        t_full -= waveform(t_full, TXc).nSigmaMean()[0]
        RX=dict()
        for row, r_val in enumerate(h5f['r_eff']):
            rx0=h5f['p'][row,:]            
            temp=np.convolve(TX.p.ravel(), rx0, 'full')*0.25e-9
            RX[r_val]=waveform(TX.t, np.interp(TX.t.ravel(), t_full.ravel(), temp).reshape(TX.t.shape))
            RX[r_val].t0=0.
            RX[r_val].tc=RX[r_val].nSigmaMean()[0]
    return RX
            
            
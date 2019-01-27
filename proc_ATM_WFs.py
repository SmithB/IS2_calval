#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:53:48 2019

@author: ben

minimal code to process a few waveforms for plotting.  Not optimized for large-scale processing.


"""

import numpy as np
from read_ATM_wfs import read_ATM_file
from fit_waveforms import fit_catalog
#from fit_waveforms import waveform
from make_rx_scat_catalog import make_rx_scat_catalog
import h5py

np.seterr(invalid='ignore')

def proc_WFs(WF_file, TX_file,   shots, scat_file=None):
    
    h5f=h5py.file(TX_file)
    TX={'t':np.array(h5f['/TX/t']),'p':np.array(h5f(['/TX/p']))}
    h5f.close()
    
    # make the library of templates
    WF_library = dict()
    WF_library.update({0.:TX})     
    WF_library.update(make_rx_scat_catalog(TX, h5_file=scat_file))
     
    catalogBuffer = None
    result=dict()
    for shot0 in shots:
        D=read_ATM_file(WF_file, shot0=shot0, nShots=1)
        
        # make the return waveform structure
        rxData=D['RX'][0:D['RX'].size]
        rxData.t=rxData.t-rxData.t.mean()
        nP=np.ones(rxData.size)
        sigmas=np.arange(0, 5, 0.125)
        rxData.nPeaks=nP
        threshold_mask = rxData.p >= 255
        rxData.subBG()
        rxData.tc=[ ii.nSigmaMean()[0] for ii in rxData ]
        rxData.p[threshold_mask]=np.NaN
        
        # choose a set of delta t values
        deltas=np.arange(-1, 1.5, 0.5)  
            
        # fit the data. Catalogbuffer contains waveform templates that have already been tried
        D_out, catalogBuffer=fit_catalog(rxData, WF_library, sigmas, deltas, return_data_est=True, \
                                         return_catalog=True, catalog=catalogBuffer)
        for ind, shot in enumerate(D_out['shot']):
            if shot in shots:
                result[shot]=dict()
                for key in D_out:
                    result[shot][key]=D_out[key][ind]
    return result
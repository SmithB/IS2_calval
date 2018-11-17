#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:51:42 2018

@author: ben
"""

import numpy as np
#import matplotlib
#matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from read_ATM_wfs import read_ATM_file, est_mean_wf, normalize_wf
from fit_waveforms import fit_library

fname='/data/ATM_WF/ramp_passes/2017.12.04/ILNSAW1B_20171204_172637.atm6BT7.caltableApplied.h5'

try:
    print(D['TX'].shape)
except: 
    D=read_ATM_file(fname, shots=np.arange(5000, dtype=int))
    D['TXn']=np.zeros_like(D['TX'], dtype=np.float64)
    for col in range(D['TX'].shape[1]):
        D['TXn'][:,col]=normalize_wf(D['TX'][:,col])
if True:
    plt.figure()
    plt.imshow(D['TX'], aspect='auto')
    plt.title('transmit waveform')
#plt.imshow(temp[:, (temp==255).sum(axis=0)==0])

if False:
    plt.figure(); 
    R=np.mean(D['TX'][90:100,:].astype(np.float64), axis=0)
    plt.plot(R*np.cos(D['az']*np.pi/180), R*np.sin(D['az']*np.pi/180.),'.', markersize=0.5)
    plt.title('power afn azimuth')

TXm, tx_sigma=est_mean_wf(D['TX'])
if True:
    plt.figure()
    plt.plot(TXm,'k')
    plt.plot(TXm-tx_sigma,'r--')
    plt.plot(TXm+tx_sigma,'r--')

misfit=np.zeros(D['TX'].shape[1])
for col in range(D['TX'].shape[1]):
    wf_norm=normalize_wf(D['TX'][:,col])
    misfit[col]=np.sum((wf_norm-TXm)**2)

if False:
    plt.plot(normalize_wf(D['TX'][:,0]),'g')

if False:
    WF_library={'avg':{'t':np.arange(TXm.size),'p':TXm}}
    deltas=np.arange(-2, 2.1, 0.1)
    sigmas=np.arange(0, 5, 0.25)
    D1s=list()
    for ind in [0, 1, 30]:
        temp=D['TX'][:,ind].astype(np.float64)
        temp[temp==255]=np.NaN
        D1s.append({'t':np.arange(temp.size),'p':temp})
    fit_library(D1s, WF_library, sigmas, deltas)
            

#plt.figure()
#plt.plot(misfit)
    



#if False:
#    D_in=h5py.File(fname,'r')
#    sg0=list()
#    for shot in range(n_shots):
#        gate0=D_in['/waveforms/twv/shot/gate_start'][shot]
#        gateN=gate0+D_in['/waveforms/twv/shot/gate_count'][shot]
#        gates=np.arange(gate0, gateN)
#        pos=D_in['/waveforms/twv/gate/position'][gates]
#        sg0.append({'shot':np.zeros_like(gates)+shot,'pos':pos})
#    
#    plt.figure()
#    plt.plot(np.concatenate([x['shot'] for x in sg0]), np.concatenate([x['pos'] for x in sg0]),'.')
#    plt.show()

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
from fit_waveforms import fit_library, nSigmaMean, shift_vector

fname='/data/ATM_WF/ramp_passes/2017.12.04/ILNSAW1B_20171204_172637.atm6BT7.caltableApplied.h5'

try:
    print(D['TX'].shape)
except: 
    D=read_ATM_file(fname, shots=np.arange(5000, dtype=int))
    D['TXn']=np.zeros_like(D['TX'], dtype=np.float64)
    for col in range(D['TX'].shape[1]):
        D['TXn'][:,col]=normalize_wf(D['TX'][:,col])
if False:
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
if False:
    plt.figure()
    plt.plot(TXm,'k')
    plt.plot(TXm-tx_sigma,'r--')
    plt.plot(TXm+tx_sigma,'r--')

misfit=np.zeros(D['TX'].shape[1])
for col in range(D['TX'].shape[1]):
    wf_norm=normalize_wf(D['TX'][:,col])
    misfit[col]=np.sum((wf_norm-TXm)**2)

# calculate the mean of the WFs most similar to the mean
TXm, tx_sigma=est_mean_wf(D['TX'][:, misfit<0.1])
t0=np.arange(D['TX'].shape[0])*D['dt']
txCtr, txSigma=nSigmaMean({'t':t0, 'p':TXm})

if False:
    plt.plot(t0, TXm); plt.plot(t0-5.05625, shift_vector(t0-5.05625, t0, TXm))
    vv=shift_vector(t0-5.05625, t0, TXm)
    vi=np.interp(t0-5.05625, t0, TXm)
    plt.figure()
    plt.plot(t0, vv-vi)

if True:
    WF_library={'avg':{'t':t0-txCtr,'p':TXm}}
    deltas=np.arange(-5, 6, 0.5)
    sigmas=np.arange(0, 5, 0.25)
    D1s=list()
    for ind in range(1000):
        temp=D['RX'][:,ind].astype(np.float64)
        temp[temp==255]=np.NaN
        D1s.append({'t_start':txCtr-t0[0], 't_samp':D['dt'],'p':temp})
    wfP=fit_library(D1s, WF_library, sigmas, deltas, return_data_est=False)
    plt.figure(); 
    plt.subplot(311)
    plt.plot([ii['A'] for ii in wfP],'.')
    plt.subplot(312)
    plt.plot([ii['R']/ii['A'] for ii in wfP],'.')
    plt.subplot(313)
    plt.plot([ii['sigma'] for ii in wfP],'.')
    
        

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

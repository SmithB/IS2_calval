#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:51:42 2018

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
from read_ATM_wfs import read_ATM_file
from fit_waveforms import fit_catalog
#from fit_waveforms import waveform
from make_rx_scat_catalog import make_rx_scat_catalog
from waveform import waveform
import scipy.stats as sps
import copy
import argparse
import h5py
import os

def get_tx_est(filename, nShots=None):
    # get the transmit pulse mean
    D=read_ATM_file(filename, nShots=nShots, readTX=True, readRX=False)
    
    TXm=D['TX'].calcMean().normalize()
    misfit=np.zeros(D['TX'].size)
    TXn=D['TX'][np.arange(0, D['TX'].size, dtype=int)].normalize()
    misfit=np.sqrt(np.mean((TXn.p-TXm.p)**2, axis=0))
    
    # calculate the mean of the WFs most similar to the mean
    TX0 = D['TX'][misfit<2*np.median(misfit)].calcMean()
    txC, txSigma=TX0.nSigmaMean()
    TX0.t=TX0.t-txC
    TX0.tc=0
    
    deltas = np.arange(-2.5, 2.5, 0.5)
    sigmas = np.arange(0, 1, 0.25)
    
    # cleanup the input transmit data
    txData = D['TX'][0:D['TX'].size]
    txData.t=txData.t-txData.t.mean()
    thresholdMask = txData.p >= 255
    txData.subBG()
    txData.tc=np.array([txData[ii].nSigmaMean()[0] for ii in range(txData.size)])
    txData.p[thresholdMask]=np.NaN
    txData.nPeaks=np.ones(txData.size)
    
    # minimize the shifted misfit between each transmit pulse and the waveform mean
    txP=fit_catalog(txData, {0.:TX0}, sigmas, deltas)
    
    # evaluate the fit and find the waveforms that best match the mean transmitted pulse
    RR = txP['R'] / txP['A']
    error_tol=sps.scoreatpercentile(RR, 68)
    all_shifted_TX=list()
    for ii in range(len(txP['R'])):
        temp=np.interp(TX0.t.ravel(), txData.t.ravel() - txP['delta_t'][ii], txData[ii].p.ravel())
        if RR[ii] < error_tol and txP['A'][ii] < 250: 
            temp = (temp - txP['B'][ii]) / txP['A'][ii]
            all_shifted_TX.append(temp)
    # put together the shifted transmit pulses that passed muster        
    all_shifted_TX = np.c_[all_shifted_TX]
    TX = copy.copy(TX0)
    TX.p = np.nanmean(all_shifted_TX, axis=0).reshape(TX0.t.shape)
    TX.p = np.interp(TX.t.ravel(), TX.t.ravel()-TX.nSigmaMean()[0], TX.p.ravel()).reshape(TX.t.shape)
    TX.tc = np.array(TX.nSigmaMean()[0])
    TX.normalize()
    return TX
    

parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters')
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
parser.add_argument('--startShot', '-s', type=int, default=0)
parser.add_argument('--nShots', '-n', type=int, default=np.Inf)
parser.add_argument('--DOPLOT', '-P', action='store_true')
parser.add_argument('--waveforms', '-w', action='store_true', default=False)
args=parser.parse_args()

# get the waveform count from the output file 
nWFs=np.minimum(args.nShots, read_ATM_file(args.input_file, getCountAndReturn=True)-args.startShot)
lastShot=args.startShot+nWFs

# make the output file
if os.path.isfile(args.output_file):
    os.remove(args.output_file)

outDS = ['shot', 'R', 'A', 'B', 'delta_t', 'sigma', 't0', 'K0', 'Kmin', 'Kmax',\
        'latitude', 'longitude', 'elevation']
out_h5 = h5py.File(args.output_file,'w')
for DS in outDS:
    out_h5.create_dataset(DS, (nWFs,))

if args.waveforms:
    out_h5.create_dataset('RX/p', (192, nWFs))
    out_h5.create_dataset('RX/p_fit', (192, nWFs))

# choose how to divide the output
blocksize=1000
start_vals=args.startShot+np.arange(0, nWFs, blocksize, dtype=int)
 
# get the transmit pulse
TX = get_tx_est(args.input_file)

# make the library of templates
WF_library = dict()
WF_library.update({0.:TX})
WF_library.update(make_rx_scat_catalog(TX))
R_vals=np.sort(list(WF_library))

print("Returns:")
# loop over start vals (one block at a time...)
catalogBuffer=None
for shot0 in start_vals:
    D=read_ATM_file(args.input_file, shot0=shot0, nShots=np.minimum(blocksize, lastShot-shot0))
    
    # make the return waveform structure
    rxData=D['RX'][0:D['RX'].size]
    rxData.t=rxData.t-rxData.t.mean()
    nP=rxData.count_peaks(W=4)
    rxData.nPeaks=nP
    threshold_mask = rxData.p >= 255
    rxData.subBG()
    rxData.tc=[ ii.nSigmaMean()[0] for ii in rxData ]
    rxData.p[threshold_mask]=np.NaN
    
    # choose a set of delta t values and sigma values
    deltas=np.arange(-2, 2.5, 0.5)
    sigmas=np.arange(0, 5, 0.25)
        
    # fit the data
    D_out, catalogBuffer=fit_catalog(rxData, WF_library, sigmas, deltas, return_data_est=args.waveforms, \
                                     return_catalog=True, catalog=catalogBuffer)
    
    N_out=D_out['shot'].size
    outShot0=shot0-args.startShot
    for key in outDS:
        if key in D_out:
            out_h5[key][outShot0:outShot0+N_out]=D_out[key]
        else:
            out_h5[key][outShot0:outShot0+N_out]=D[key]
    
    if args.waveforms:
        out_h5['RX/p'][:, outShot0:outShot0+N_out] = D_out['wf_est']
        out_h5['RX/p_fit'][:, outShot0:outShot0+N_out] = rxData.p

    if args.DOPLOT:
        plt.figure(); 
        plt.subplot(511)
        plt.plot( D_out['A'], '.')
        plt.ylabel('amplitude')
        plt.subplot(512)
        plt.plot(D_out['R']/D_out['A'], '.')
        plt.ylabel('R/A')
        plt.subplot(513)
        plt.plot( D_out['sigma'], '.')
        plt.ylabel('sigma')
        plt.subplot(514)
        plt.plot( D_out['delta_t'], '.')
        plt.ylabel('delta_t')
        plt.subplot(515)
        plt.plot( D_out['K0'], '.')
        plt.plot( D_out['Kmax'], '.')
        plt.ylabel('K')

out_h5.create_dataset("TX/t", data=TX.t.ravel())
out_h5.create_dataset("TX/p", data=TX.p.ravel())
if args.waveforms:
    out_h5.create_dataset('RX/t', data=rxData.t.ravel())
    
out_h5.close() 
        
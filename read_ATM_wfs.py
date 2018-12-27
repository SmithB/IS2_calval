#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:31:16 2018

@author: ben
"""
import h5py
import numpy as np
#import matplotlib
#matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from waveform import waveform

def read_wf(D, shot, starting_sample=0, read_tx=False, read_rx=False, read_all=False):
    """
    read transmit and receive pulses for a shot from data extracted from an h5 file
    """
    gate0=D['/waveforms/twv/shot/gate_start'][shot]-1
    gateN=gate0+D['/waveforms/twv/shot/gate_count'][shot]
    result=list()
    result=dict()
    if read_tx:
        result['tx']={'gate':gate0+D['/laser/gate_xmt'][shot]-1}
    if read_rx:
        result['rx']={'gate':gate0+D['/laser/gate_rcv'][shot]-1}
    if read_all:
        for ii in np.arange(gateN-gate0+1, dtype=int):
            result[ii]={'gate':gate0+ii}
    for key in result:
        gate=result[key]['gate']
        samp0=D['/waveforms/twv/gate/wvfm_start'][gate]-1-starting_sample
        sampN=samp0+D['/waveforms/twv/gate/wvfm_length'][gate]
        result[key].update({'pos':D['/waveforms/twv/gate/position'][gate],\
              'P':D['/waveforms/twv/wvfm/amplitude'][samp0:sampN],\
              'count':D['/waveforms/twv/gate/pulse/count'][gate]})
    return result

def read_ATM_file(fname, getCountAndReturn=False, shot0=0, nShots=np.Inf, readTX=True, readRX=True):
    """
    Read data from an ATM file
    """
    with h5py.File(fname,'r') as h5f:
        
        # figure out what shots to read
        shotMax=h5f['/waveforms/twv/shot/gate_start'].size
        if getCountAndReturn:
            return shotMax
        
        nShots=np.minimum(shotMax-shot0, nShots)        
        shotN=np.int(shot0+nShots)
        shot0=np.int(shot0)
        # read in some of the data fields
        D_in=dict()

        # read the waveform starts, stops, and lengths for all shots in the file (inefficient, but hard to avoid)
        for key in ('/waveforms/twv/gate/wvfm_start', '/waveforms/twv/gate/wvfm_length', '/waveforms/twv/gate/position',\
                    '/waveforms/twv/gate/pulse/count'):
            D_in[key]=np.array(h5f[key], dtype=int)
        # read in the gate info for the shots we want to read    
        for key in( '/waveforms/twv/shot/gate_start', '/waveforms/twv/shot/gate_count', '/laser/gate_xmt', '/laser/gate_rcv'):
            D_in[key]=np.array(h5f[key][shot0:shotN], dtype=int)
            
        #read in the geolocation
        try:
            for key in ('footprint/latitude','footprint/longitude','footprint/elevation','/laser/scan_azimuth'):
                D_in[key]=np.array(h5f[key][shot0:shotN])
        except KeyError:
            pass
        # read the sampling interval
        dt=np.float64(h5f['/waveforms/twv/ancillary_data/sample_interval'])
        
        # figure out what samples to read from the 'amplitude' dataset
        gate0=D_in['/waveforms/twv/shot/gate_start'][0]-1 + D_in['/laser/gate_xmt'][0]-1
        sample_start = D_in['/waveforms/twv/gate/wvfm_start'][gate0]-1
        gateN = D_in['/waveforms/twv/shot/gate_start'][-1]-1 + D_in['/laser/gate_rcv'][-1]-1
        sample_end =  D_in['/waveforms/twv/gate/wvfm_start'][gateN] + D_in['/waveforms/twv/gate/wvfm_length'][gateN]
        # ... and read the amplitude.  The sample_start variable will get subtracted off
        # subsequent indexes into the amplitude array
        key='/waveforms/twv/wvfm/amplitude'
        D_in[key]=np.array(h5f[key][sample_start:sample_end+1], dtype=int)
        
        
        TX=list()
        RX=list()
        tx_samp0=list()
        rx_samp0=list()
        RX=list()
        nPeaks=list()
        rxBuffer=np.zeros(192)+np.NaN
        for shot in range(int(nShots)):
            wfd=read_wf(D_in, shot, starting_sample=sample_start, read_tx=readTX, read_rx=readRX)
            if readTX:
                TX.append(wfd['tx']['P'][0:160])
                tx_samp0.append(wfd['tx']['pos'])
            if readRX:  
                nRX=np.minimum(190, wfd['rx']['P'].size)
                rxBuffer[0:nRX]=wfd['rx']['P'][0:nRX]
                rxBuffer[nRX+1:-1]=np.NaN
                RX.append(rxBuffer.copy())
                nPeaks.append(wfd['rx']['count'])
                rx_samp0.append(wfd['rx']['pos'])
        shots=np.arange(shot0, shotN, dtype=int)
        try:
            result={ 'az':D_in['/laser/scan_azimuth'],'dt':dt, 'elevation':D_in['footprint/elevation'], 'latitude':D_in['footprint/latitude'],'longitude':D_in['footprint/longitude']}
        except KeyError:
            result={}
        if readTX:
            TX=np.c_[TX].transpose() 
            result['TX']=waveform(np.arange(TX.shape[0])*dt, TX, shots=shots)
            
        if readRX:
            RX=np.c_[RX].transpose()
            nPeaks=np.c_[nPeaks].ravel()
            result['RX']=waveform(np.arange(RX.shape[0])*dt, RX, shots=shots, nPeaks=np.c_[nPeaks.ravel()])
        
    return result

def normalize_wf(wf, noise_samps=[0, 30]):
    """
    normalize an input waveform to have a peak of 1 and a pre-trigger mean of zero
    """
    P=wf.astype(np.float64)
    bg=np.mean(P[noise_samps[0]:noise_samps[1]])
    A=P.max()-bg
    P=(P-bg)/A
    return P

def est_mean_wf(P):
    """
    Estimate the mean waveform from a collection
    """
    notSaturated=np.where(((P==255).sum(axis=0)<2) & (np.all(np.isfinite(P), axis=0)))[0]
    P1=np.zeros((P.shape[0], notSaturated.size))
    for col_out, col in enumerate(notSaturated):
        P1[:,col_out]=normalize_wf(P[:,col])
    good=np.mean(P1[115:125,:], axis=0)<0.1
    wf_bar=np.mean(P1[:,good], axis=1)
    wf_sigma=np.std(P1[:,good], axis=1)
    return wf_bar, wf_sigma
    

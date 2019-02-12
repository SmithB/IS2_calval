#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:51:42 2018

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
from IS2_calval.read_ATM_wfs import read_ATM_file
from IS2_calval.fit_waveforms import fit_catalog
#from fit_waveforms import waveform
from IS2_calval.make_rx_scat_catalog import make_rx_scat_catalog
from IS2_calval.waveform import waveform
from time import time
import scipy.stats as sps
import copy
import argparse
import h5py
import os

np.seterr(invalid='ignore')
os.environ["MKL_NUM_THREADS"]="1"  # multiple threads don't help that much

def get_tx_est(filename, nShots=np.Inf, TX0=None, source='TX', pMin=50, skip_n_tx=None, skip_make_TX=False):
    if source is 'TX':
        # get the transmit pulse mean
        D=read_ATM_file(filename, nShots=nShots, readTX=True, readRX=False)      
        TX=D['TX']        
    if source is 'RX':
        D=read_ATM_file(filename, nShots=nShots, readRX=True, readTX=False)
        TX=D['RX']
    if TX0 is None:
        # select waveforms that are not clipped and have adequate amplitude
        valid=np.where((np.nanmax(TX.p,axis=0) < 255) & (np.nanmax(TX.p,axis=0) > pMin) & (D['calrng'] > 5))[0]
        TX1=TX[valid]
        t50=TX1.t50()
        ti=TX1.t.ravel()-np.mean(TX1.t)
        # align the pulses on their 50% threshold
        for ii in range(TX1.size):
            TX1.p[:,ii]=np.interp(ti, TX1.t.ravel()-t50[ii], TX1.p[:,ii])
        TXm=TX1.calcMean().subBG().normalize()
        TXn=TX1[np.arange(0, TX1.size, dtype=int)].subBG().normalize()
        misfit=np.sqrt(np.nanmean((TXn.p-TXm.p)**2, axis=0))
        
        # calculate the mean of the WFs most similar to the mean
        TX0 = TXn[misfit<2*np.median(misfit)].calcMean()
        txC, txSigma=TX0.nSigmaMean()
        TX0.t=TX0.t-txC
        TX0.tc=0
    
    # Prepare the input txdata for fitting
    # Use a subsetting operation to copy the data and remove clipped WFs
    calrng=D['calrng']
    if skip_n_tx is not None:
        ind=np.arange(0, TX.size, skip_n_tx, dtype=int)
        TX=TX[ind]
        calrng=calrng[ind]        
    valid=np.where((np.nanmax(TX.p,axis=0) < 255) & (np.nanmax(TX.p,axis=0) > pMin) & (calrng>1))[0]
    txData = TX[valid]
    txData.t=txData.t-txData.t.mean()
    thresholdMask = txData.p >= 255
    txData.subBG()
    txData.tc=np.array([txData[ii].nSigmaMean()[0] for ii in range(txData.size)])
    txData.p[thresholdMask]=np.NaN
    txData.nPeaks=np.ones(txData.size)
    
    t_old=time() 
    deltas = np.arange(-2.5, 2.5, 1)
    sigmas = np.arange(0, 1.5, 0.25)
    # minimize the shifted misfit between each transmit pulse and the waveform mean
    txP=fit_catalog(txData, {0.:TX0}, sigmas, deltas)
    print("     time to fit start pulse=%3.3f" % (time()-t_old))
    
    if skip_make_TX is True:
        return dict(), txP
    
    # evaluate the fit and find the waveforms that best match the mean transmitted pulse
    RR = txP['R'] / txP['A']
    error_tol=sps.scoreatpercentile(RR, 68)
    all_shifted_TX=list()
    for ii in range(len(txP['R'])):
        temp=np.interp(TX0.t.ravel(), txData.t.ravel() - txP['delta_t'][ii], txData[ii].p.ravel())
        if RR[ii] < error_tol and txP['A'][ii] < 250 and txP['sigma'][ii] <= sigmas[2]: 
            temp = (temp - txP['B'][ii]) / txP['A'][ii]
            all_shifted_TX.append(temp)
    # put together the shifted transmit pulses that passed muster        
    all_shifted_TX = np.c_[all_shifted_TX]
    TX = copy.copy(TX0)
    TX.p = np.nanmean(all_shifted_TX, axis=0).reshape(TX0.t.shape)
    TX.p = np.interp(TX.t.ravel(), TX.t.ravel()-TX.nSigmaMean()[0], TX.p.ravel()).reshape(TX.t.shape)
    TX.tc = np.array(TX.nSigmaMean()[0])
    TX.normalize()
    return TX, txP

def proc_RX(WF_file, shots, rxData=None, sigmas=np.arange(0, 5, 0.25), deltas=np.arange(-1, 1.5, 0.5), TX=None, countPeaks=True, WF_library=None, TX_file=None, scat_file=None, catalogBuffer=None):
    """
        Routine to process the wavefoms in an ATM waveform file.  Can be run inside 
        the loop of fit_ATM_scat, or run on a few waveforms at a time for plot generation
    """
    if TX is None and TX_file is not None:
        with h5py.file(TX_file) as h5f:
            TX=waveform(np.array(h5f['/TX/t']),np.array(h5f(['/TX/p'])))
    TX.t -= TX.nSigmaMean()[0]
    TX.tc = 0
            
        
    # make the library of templates
    if WF_library is None:
        WF_library = dict()
        WF_library.update({0.:TX}) 
        if scat_file is not None:
            WF_library.update(make_rx_scat_catalog(TX, h5_file=scat_file))
    if rxData is None: 
        # make the return waveform structure
        D=read_ATM_file(WF_file, shot0=shots[0], nShots=shots[-1]-shots[0])
        rxData=D['RX']
        rxData.t -= np.nanmean(rxData.t)
    else:
        D=dict()
    if countPeaks:
        rxData.nPeaks=rxData.count_peaks(W=4)
    else:
        rxData.nPeaks=np.ones(rxData.size)

    threshold_mask = rxData.p >= 255
    rxData.subBG()
    rxData.tc=[ ii.nSigmaMean()[0] for ii in rxData ]
    rxData.p[threshold_mask]=np.NaN
             
    # fit the data. Catalogbuffer contains waveform templates that have already been tried
    D_out, catalogBuffer=fit_catalog(rxData, WF_library, sigmas, deltas, return_data_est=True, \
                                     return_catalog=True, catalog=catalogBuffer)
    return D_out, rxData, D, catalogBuffer


def main():
    parser = argparse.ArgumentParser(description='Fit the waveforms from an ATM file with a set of scattering parameters')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--startShot', '-s', type=int, default=0)
    parser.add_argument('--scat_file', '-f', type=str, default=None)
    parser.add_argument('--nShots', '-n', type=int, default=np.Inf)
    parser.add_argument('--DOPLOT', '-P', action='store_true')
    parser.add_argument('--IR', '-I', action='store_true')
    parser.add_argument('--TXfromRX', action='store_true')
    parser.add_argument('--skipRX', action='store_true', default=False)
    parser.add_argument('--fitTX', action='store_true')
    parser.add_argument('--everyNTX', type=int, default=100)
    parser.add_argument('--TXfile', '-T', type=str, default=None)
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
        out_h5.create_dataset(DS, (nWFs,), dtype='f8')
    
    if args.waveforms:
        out_h5.create_dataset('RX/p', (192, nWFs))
        out_h5.create_dataset('RX/p_fit', (192, nWFs))

    TxP=None
    # get the transmit pulse
    if args.TXfile is not None:
        with h5py.File(args.TXfile,'r') as fh:
            TX=waveform(np.array(fh['/TX/t']), np.array(fh['/TX/p']) )
        TX.t -= TX.nSigmaMean()[0]
        TX.tc = np.array(TX.nSigmaMean()[0])
        TX.normalize()
    else:     
        if args.TXfromRX is False:
            TX, TxP = get_tx_est(args.input_file, nShots=5000)
        else:
            TX, TxP = get_tx_est(args.input_file, source='RX', pMin=150)
    if args.fitTX:
        print('fitting the transmit pulse')        
        dummy, TxP = get_tx_est(args.input_file, source='TX', TX0=TX, skip_make_TX=True, skip_n_tx=args.everyNTX)

    out_h5.create_dataset("TX/t", data=TX.t.ravel())
    out_h5.create_dataset("TX/p", data=TX.p.ravel())

    if TxP is not None:
        for field in ['t0','A','R','shot','sigma']:
            out_h5.create_dataset('/TX/'+field, data=TxP[field])

    if args.skipRX is True:
        out_h5.close()
        return

    # check whether the scattering file exists
    if args.scat_file is not None:
        scat_file=args.scat_file
    else:
        # look for the default scattering file in the directory where the source file is found
        scat_file=os.path.dirname(os.path.abspath(__file__)) + '/subsurface_srf_no_BC.h5'
    if not os.path.isfile(scat_file):
        print("%s does not exist" % scat_file)
        exit()

    # make the library of templates
    WF_library = dict()
    WF_library.update({0.:TX})
    if args.IR is False:
        WF_library.update(make_rx_scat_catalog(TX, h5_file=scat_file))
        
    print("Returns:")
    # loop over start vals (one block at a time...)
    # choose how to divide the output
    blocksize=1000
    start_vals=args.startShot+np.arange(0, nWFs, blocksize, dtype=int)

    catalogBuffer=None
    time_old=time()
    
    if args.IR is True:
        countPeaks=False
        sigmas=np.arange(0, 5, 0.125)
    else:
        countPeaks=True   
        sigmas=np.arange(0, 5, 0.25)
    # choose a set of delta t values
        deltas=np.arange(-1, 1.5, 0.5)  
    
    for shot0 in start_vals:
        shots=np.arange(shot0, np.minimum(shot0+blocksize, lastShot))
        D_out, rxData, D, catalogBuffer=proc_RX(args.input_file, shots, sigmas=sigmas, deltas=deltas,\
             TX=TX, WF_library=WF_library, catalogBuffer=catalogBuffer, countPeaks=countPeaks)
             
        N_out=D_out['shot'].size
        outShot0=shot0-args.startShot
        for key in outDS:          
            try:
                if key in D_out:
                    out_h5[key][outShot0:outShot0+N_out]=D_out[key].ravel()
                else:
                    if key in D:
                        out_h5[key][outShot0:outShot0+N_out]=D[key].ravel()
            except OSError:
                print("OSError for key=%s, outshot0=%d, outshotN=%d, nDS=%d"% (key, outShot0, outShot0+N_out, out_h5[key].size))
        
        if args.waveforms:
            out_h5['RX/p_fit'][:, outShot0:outShot0+N_out] = D_out['wf_est']
            out_h5['RX/p'][:, outShot0:outShot0+N_out] = rxData.p
        
        print("  shot=%d out of %d, N_keys=%d" % (shot0+blocksize, start_vals[-1]+blocksize, len(catalogBuffer.keys())))
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
    print("   time to fit RX=%3.2f" % (time()-time_old))
    
    if args.waveforms:
        out_h5.create_dataset('RX/t', data=rxData.t.ravel())
        
    out_h5.close() 
            
if __name__=="__main__":
    main()
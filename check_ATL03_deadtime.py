#! /usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Calculate the differenes between the channel biases in the two channels in each receiver pixel from an ATL03 file

This script reads and ATL03 file, and uses the inter-photon arrival times to deduce
the difference between the timing biases for the two channels in each detector pixel.

For each pixel in the detector, it calculates the apprent deadtime for pairs 
of photons for which the higher-numbered channel comes first, and the apparent 
deadtime for pairs photons for which the lower-numbered channel comes first.  
The difference between the two apparent deadtimes gives the difference in 
timing bias between the two channels.

Input (one argument):  A glob string or filename to define the input ATL03 files (add quotes if including wildcards)

Output (one per channel per file): a list of numbers:
    Channel # for first channel in the pixel
    number of delta-t values in the pixel
    fraction of all photons in the pixel
    number of greater-lesser deltas in the pixel
    number of lesser-greater deltas in the pixel
    deadtime from greater-lesser differneces
    deadtime from lesser-greater differences


Created on Sun Oct 14 19:33:45 2018

@author: Ben Smith
"""

import h5py
import numpy as np
#import matplotlib.pyplot as plt
import glob
import sys

def deadtime_from_hist(t, counts, threshold, t_min):
    """
    Calculate the apparent deadtime from an input histogram
    
    Inputs:
        t: the center times for the histogram bins
        counts: the number of photons in each bin
        threshold: The fraction of the maximum bin used in identifying bins 
            containing signal
        t_min : the minumum time value for bins in the histogram over which signal might be 
            found (helps ignore the spike that can appear at zero for spurious double counts)
    Outputs:
        tHalf: the first bin for which the count reaches half the mean value of 
            all the counts after the maximum value
    """
    t_gt=t[t>t_min]
    t_max=t_gt[np.argmax(counts[t>t_min])]
    Cthr=np.mean(counts[t>t_max])*threshold
    tHalf=np.min(t[(counts>Cthr) & (t>t_min)])
    return tHalf

# find all files that match the input glob string (also works with just one file)
files=glob.glob(sys.argv[1])
fields=('h_ph','ph_id_pulse','pce_mframe_cnt','ph_id_channel');

# beams:
beams=('gt1l','gt1r','gt2l','gt2r','gt3l','gt3r')
c2=1.5e8  # half speed of light, m/s
dt_hist=0.05*1.e-9  # time increment of histograms, s
t_hist=np.arange(-1e-9, 1.e-8, dt_hist)  # time of histogram, s
bin_ctrs=t_hist[0:-1]+dt_hist/2.  # times for histogram centers

D=dict()
for thefile in files:
    dt_est_minus=dict()
    dt_est_plus=dict()
    with h5py.File(thefile,'r') as f:
        delta_dead_est=dict()
        
        for beam in beams:
            # read in the data
            for field in fields:
                D[field]=np.array(f[beam]['heights'][field], dtype=np.float64)
            # identify the channels that are present in this beam
            channels=np.unique(D['ph_id_channel'])
            n_px=int(channels.size/2)
             
            # loop over pixels (identified by the first channel in the pixel)
            for channel in channels[0:n_px]:
                Dch=dict()
                # find the photons in the current pixel (this channel number, or +60)
                ii=np.where(np.logical_or(D['ph_id_channel']==channel, D['ph_id_channel']==channel+60))[0]
                for field in fields:
                    Dch[field]=D[field][ii]
                # calculate the pulse number
                Dch['pulse_num']=(Dch['pce_mframe_cnt']-Dch['pce_mframe_cnt'].min())*200.+Dch['ph_id_pulse']
                # sort photons by pulse number
                ii=np.argsort(Dch['pulse_num'])
                for field in fields:
                    Dch[field]=Dch[field][ii]
                # loop over pulses
                pulses, ip=np.unique(Dch['pulse_num'], return_index=True)
                # initialize output lists to accumulate photon-to-photon time differences
                delta_t=list()
                delta_ch=list()
                for ind in np.arange(pulses.size-1):
                    if ip[ind+1]==ip[ind]+1:
                        # skip if there's only one return from this pulse
                        continue
                    # sort photons in the pulse by time
                    ind_h=ip[ind]+np.argsort(Dch['h_ph'][ip[ind]:ip[ind+1]])
                    # append the delta times between the sorted photons to the list
                    delta_t.append(np.diff(Dch['h_ph'][ind_h]/c2))
                    delta_ch.append(np.diff(Dch['ph_id_channel'][ind_h]))
                # concatenate the output lists into numpy arrays 
                delta_t=np.array(np.concatenate(delta_t))
                delta_ch=np.array(np.concatenate(delta_ch))
                
                # make a histogram of the delta times for which the first channel number is smaller than the second
                counts, bins=np.histogram(delta_t[delta_ch<0], t_hist)
                # ... and calculate the effective deadtime
                dt_est_minus[channel]=deadtime_from_hist(bin_ctrs, counts, 0.5, 0.75e-9)  
                N_plus=np.sum(counts)
                # same for the delta times for which the first channel number is greater than the second
                counts, bins=np.histogram(delta_t[delta_ch>0], t_hist)                                                          
                dt_est_plus[channel]=deadtime_from_hist(bin_ctrs, counts, 0.5, 0.75e-9)  
                N_minus=np.sum(counts)
                # output : Channel #, number of delta-t values, fraction of photons in this pixel, number of greater-lesser deltas, number of lesser-greater deltas, deadtime from greater-lesser, deadtime from lesser-greater
                print("%d\t%d\t%3.3f\t%d\t%d\t%3.2f\t%3.2f" % (channel, delta_t.size, Dch['h_ph'].size/np.float(D['h_ph'].size)*n_px, N_minus, N_plus, dt_est_minus[channel]*1.e9, dt_est_plus[channel]*1.e9))
 
    
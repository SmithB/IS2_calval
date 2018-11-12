#! /usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:33:45 2018

@author: ben
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def deadtime_from_hist(t, counts, threshold, t_min):
    t_gt=t[t>t_min]
    t_max=t_gt[np.argmax(counts[t>t_min])]
    Cthr=np.mean(counts[t>t_max])*threshold
    tHalfMax=np.min(t[(counts>Cthr) & (t>t_min)])
    return tHalfMax


#thefile='/Volumes/ice2/ben/scf/plateau_03/ATL03_2018_10_02_00100110_944_01.h5'
files=glob.glob(sys.argv[1])
fields=('h_ph','ph_id_pulse','pce_mframe_cnt','ph_id_channel');

beams=('gt1l','gt1r','gt2l','gt2r','gt3l','gt3r')
c2=1.5e8
dt_hist=0.05
t_hist=np.arange(-1, 10, dt_hist)*1.e-9
bin_ctrs=t_hist[0:-1]+(t_hist[2]-t_hist[1])/2.

D=dict()
for thefile in files:
    dt_est_minus=dict()
    dt_est_plus=dict()
    with h5py.File(thefile,'r') as f:
        delta_dead_est=dict()
        
        for beam in beams:
            for field in fields:
                D[field]=np.array(f[beam]['heights'][field], dtype=np.float64)
            
            channels=np.unique(D['ph_id_channel'])
            n_px=int(channels.size/2)
             
            for channel in channels[0:n_px]:
                Dch=dict()
                ii=np.where(np.logical_or(D['ph_id_channel']==channel, D['ph_id_channel']==channel+60))[0]
                #ii=np.where(D['ph_id_channel']==channels[ch_num])
                for field in fields:
                    Dch[field]=D[field][ii]
                Dch['pulse_num']=(Dch['pce_mframe_cnt']-Dch['pce_mframe_cnt'].min())*200.+Dch['ph_id_pulse']
                ii=np.argsort(Dch['pulse_num'])
                for field in fields:
                    Dch[field]=Dch[field][ii]
                
                pulses, ip=np.unique(Dch['pulse_num'], return_index=True)
                delta_t=list()
                delta_ch=list()
                for ind in np.arange(pulses.size-1):
                    if ip[ind+1]==ip[ind]+1:
                        # skip if there's only one return from this pulse
                        continue
                    # sort by time
                    ind_h=ip[ind]+np.argsort(Dch['h_ph'][ip[ind]:ip[ind+1]])
                    delta_t.append(np.diff(Dch['h_ph'][ind_h]/c2))
                    delta_ch.append(np.diff(Dch['ph_id_channel'][ind_h]))
                delta_t=np.array(np.concatenate(delta_t))
                delta_ch=np.array(np.concatenate(delta_ch))
                
                counts, bins=np.histogram(delta_t[delta_ch<0], t_hist)
                dt_est_minus[channel]=deadtime_from_hist(bin_ctrs, counts, 0.5, 0.75e-9)  
                N_plus=np.sum(counts)
                counts, bins=np.histogram(delta_t[delta_ch>0], t_hist)                                                          
                dt_est_plus[channel]=deadtime_from_hist(bin_ctrs, counts, 0.5, 0.75e-9)  
                N_minus=np.sum(counts)
                print("%d\t%d\t%3.3f\t%d\t%d\t%3.2f\t%3.2f" % (channel, delta_t.size, Dch['h_ph'].size/np.float(D['h_ph'].size)*n_px, N_plus, N_minus, dt_est_minus[channel]*1.e9, dt_est_plus[channel]*1.e9))
 
    
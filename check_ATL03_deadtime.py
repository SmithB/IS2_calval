# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:33:45 2018

@author: ben
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
thefile='/Volumes/ice2/ben/scf/plateau_03/ATL03_2018_10_02_00100111_944_01.h5'

fields=('h_ph','ph_id_pulse','pce_mframe_cnt','ph_id_channel');

D=dict()

f=h5py.File(thefile,'r')
for field in fields:
    D[field]=np.array(f['gt1l']['heights'][field], dtype=np.float64)

channels=np.unique(D['ph_id_channel'])
n_det=int(channels.size/2)

c2=1.5e8
dt_hist=0.125
t_hist=np.arange(-1, 20, dt_hist)*1.e-9

F=np.zeros_like(channels)
N0=np.zeros_like(channels)
all_delta_t=list()
all_delta_ch=list()
for ch_num in np.arange(0, n_det, dtype=int):
    Dch=dict()
    ii=np.where(np.logical_or(D['ph_id_channel']==channels[ch_num], D['ph_id_channel']==channels[ch_num+n_det]))[0]
    #ii=np.where(D['ph_id_channel']==channels[ch_num])
    for field in fields:
        Dch[field]=D[field][ii]
    Dch['pulse_num']=(Dch['pce_mframe_cnt']-Dch['pce_mframe_cnt'].min())*200.+Dch['ph_id_pulse']
    ii=np.argsort(Dch['pulse_num'])
    for field in fields:
        Dch[field]=Dch[field][ii]
    
    pulses, ip=np.unique(Dch['pulse_num'], return_index=True)
    F[ch_num]=pulses.size/Dch['pulse_num'].size
    delta_t=list()
    delta_ch=list()
    for ind in np.arange(pulses.size-1):
        if ip[ind+1]==ip[ind]+1:
            continue
        ind_h=ip[ind]+np.argsort(Dch['h_ph'][ip[ind]:ip[ind+1]])
        delta_t.append(np.diff(Dch['h_ph'][ind_h]/c2))
        delta_ch.append(np.diff(Dch['ph_id_channel'][ind_h]))
    all_delta_t.append(np.concatenate(delta_t))
    all_delta_ch.append(np.concatenate(delta_ch))
    N0[ch_num]=float(Dch['h_ph'].size)
    print('N0=%d'% N0[ch_num])

thefig=plt.figure(); plt.clf()
for ch_num, this_delta_t in enumerate(all_delta_t):
    plt.subplot(4,1, ch_num+1)
    #thefig=plt.figure(ch_num); plt.clf();
    count, bins, patches=plt.hist(this_delta_t*1.e9, t_hist*1.e9 )
    bins=bins[0:-1]+dt_hist/2  # avoids warning about count and bin being different sizes
    print('Ch0=%d, Ch1=%d, f_single_photon=%3.4f, F_lt 0.5=%3.4f, f_2-5=%3.4f' % (channels[ch_num], channels[ch_num+n_det], 
                                                                                (N0[ch_num]-this_delta_t.size)/N0[ch_num], 
                                                                                 np.sum(count[bins<0.5])/N0[ch_num], np.sum(count[np.logical_and(bins>1, bins<5)])/N0[ch_num]))
    xl=plt.xlabel('delta t, ns')
        
    
    
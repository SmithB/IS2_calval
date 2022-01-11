#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:18:34 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from LSsurf.unique_by_rows import unique_by_rows
from PointDatabase.read_xovers import read_xovers
import scipy.stats as sps

region=sys.argv[1]
release=sys.argv[2]
cycle=sys.argv[3]
ATL06_fields=['BP','LR','W','cycle_number','rgt','h_li','h_li_sigma','x','y','spot']

#v,  delta,  bar, meta = read_xovers('/Volumes/ice2/ben/scf/AA_06/tiles/002/xovers/cycle_04')#, wildcard='E400_N-500')
v,  delta,  bar, meta = read_xovers(f'/Volumes/ice2/ben/scf/{region}_06/{release}/xovers/cycle_{cycle}')#, wildcard='E400_N-500')

# error analysis to pick out the worst RGTs:
DD={}
sign=[1, -1]
dh_dict={}

sigma=np.zeros(6)
bias=np.zeros(6)

valid=(meta['grounded']==1 ) & (np.abs(meta['slope_x']+1j*meta['slope_y'])<0.005 ) & (delta['time'].ravel() < 24*3600*30)

for spot0 in np.arange(1, 7):
    for spot1 in np.arange(1, 7):
        key=(spot0, spot1)
        ii=(v['spot'][:,0]==spot0) & (v['spot'][:,1]==spot1) & valid
        good=np.ones(ii.sum(), dtype=bool)
        this_delta=delta['h_li'][ii]
        for it in range(5):
            med=np.median(this_delta[good])
            sigma_hat=(sps.scoreatpercentile(this_delta[good], 84)-sps.scoreatpercentile(this_delta[good], 16))/2
            sigma=np.std(this_delta[good])
            good=np.abs(this_delta-med) < (3*sigma_hat)
        
        G=np.c_[(delta['time'][ii])[good]/24/3600, np.ones(good.sum())]
        m=np.linalg.solve(G.T.dot(G), G.T.dot(this_delta[good]))
        r=G.dot(m)-this_delta[good]
        
        dh_dict[key]={'med':med, 'sigma':sigma, 'sigma_hat':sigma_hat, 'N':good.sum(),'bar':m[1],'dBdt':m[0]}

        if key==(1, 6):
            plt.figure(2); plt.clf(); 
            plt.plot(G[:,0], this_delta[good]*1000,'.', markersize=0.225)
            plt.plot(G[:,0], G.dot(m)*1000,'r.')
            plt.xlabel('$\delta_t$, days')
            plt.ylabel('$\delta_h$, mm')
            plt.title('spot 1 vs. spot 6, dh = %3.1f mm + %3.1f mm/day' % (m[1]*1000, m[0]*1000))
        if key[1]==1:
            fig=plt.figure(7);
            fig.add_subplot(3,2, key[0])
            plt.hist(this_delta, np.arange(-1, 1, 0.01))
            plt.title(f'$spot_{key[1]}-spot_{key[0]}$, N={good.size}')


delta_bias=np.zeros((6,6))
sigma=np.zeros((6,6))
N=sigma.copy()
sigma_hat=sigma.copy()
bar=sigma.copy()
dBdt=sigma.copy()
mask=np.ones_like(sigma.copy())
for key in dh_dict:
    delta_bias[key[0]-1, key[1]-1]= dh_dict[key]['med']
    sigma[key[0]-1, key[1]-1]= dh_dict[key]['sigma']
    N[key[0]-1, key[1]-1]= dh_dict[key]['N']
    sigma_hat[key[0]-1, key[1]-1]= dh_dict[key]['sigma_hat']
    bar[key[0]-1, key[1]-1]= dh_dict[key]['bar']
    dBdt[key[0]-1, key[1]-1]= dh_dict[key]['dBdt']
    if key[1] < key[0]:
        mask[key[1]-1, key[0]-1]=np.NaN
    
def spot_text(val, ax, format_str):
    s1,s2 = np.meshgrid(np.arange(1,7, dtype=int), np.arange(1,7, dtype=int))
    for xyv in zip(s1.ravel(), s2.ravel(), val.ravel()):
        if np.isfinite(xyv[2]):
            ax.text(xyv[0], xyv[1], format_str % xyv[2], ha='center',va='center')
        

plt.figure(2); plt.clf()
bmax=np.max(np.abs(delta_bias.ravel()))*1000
plt.imshow(delta_bias*1000*mask, extent=[0.5, 6.5, 0.5, 6.5], cmap='Spectral_r', origin='lower', vmin=-bmax, vmax=bmax); hb=plt.colorbar();
spot_text(delta_bias.ravel()*1000*mask.ravel(), plt.gca(), '%2.0d')
hb.set_label('median bias, mm')
plt.gca().set_ylabel('early spot')
plt.gca().set_xlabel('late spot')

fig=plt.figure(3); plt.clf()
plt.imshow(sigma*1000*mask, extent=[0.5, 6.5, 0.5, 6.5], vmin=0, vmax=sigma.ravel().max()*1000, origin='lower'); hb=plt.colorbar(); 
spot_text(sigma.ravel()*1000*mask.ravel(), plt.gca(), '%2.0d')
hb.set_label('$\sigma$, mm')
plt.gca().set_ylabel('early spot')
plt.gca().set_xlabel('late spot')


plt.figure(4); plt.clf()
bmax=np.max(np.abs(bar.ravel()))*1000
plt.imshow(bar*1000*mask, extent=[0.5, 6.5, 0.5, 6.5], cmap='Spectral_r', origin='lower', vmin=-bmax, vmax=bmax); hb=plt.colorbar(); 
spot_text(bar.ravel()*1000*mask.ravel(), plt.gca(), '%2.0d')
hb.set_label('mean bias, mm')
plt.gca().set_ylabel('early spot')
plt.gca().set_xlabel('late spot')


plt.figure(5); plt.clf()
bmax=np.max(np.abs(dBdt.ravel()))*1000
plt.imshow(dBdt*1000*mask, extent=[0.5, 6.5, 0.5, 6.5], cmap='Spectral_r', origin='lower', vmin=-bmax, vmax=bmax); hb=plt.colorbar(); 
hb.set_label('bias drift, mm/day')
spot_text(dBdt.ravel()*1000*mask.ravel(), plt.gca(), '%2.2f')
plt.gca().set_xlabel('late spot')
plt.gca().set_ylabel('early spot')


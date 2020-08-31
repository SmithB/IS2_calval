# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:56:05 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from LSsurf.unique_by_rows import unique_by_rows
from PointDatabase.read_xovers import read_xovers

xover_dir=sys.argv[1]
ATL06_fields=['BP','LR','W','cycle_number','rgt','h_li','h_li_sigma','x','y','spot']

v,  delta,  bar, meta = read_xovers(xover_dir)#, wildcard='E400_N-500')
meta['slope_mag']=np.abs(meta['slope_x']+1j*meta['slope_y'])

# error analysis to pick out the worst RGTs:
DD={}
sign=[1, -1]
dh_dict={}

for col in [0, 1]:
    # loop over combinations of cycles and tracks in each column
    u_ct, D_ct = unique_by_rows(np.c_[np.round(v['rgt'][:, col]).astype(int), np.round(v['cycle_number'][:,col]).astype(int)], return_dict=True)
    for ct in u_ct:
        key=tuple(ct)
        if key not in dh_dict:
            dh_dict[key]=[]
        these=D_ct[key]
        good=(bar['atl06_quality_summary'][these]<0.01) & (meta['slope_mag'][these]<0.02)
        if np.any(good):
            dh_dict[key].append(delta['h_li'][these[good]]*sign[col])
D2={}
for key in dh_dict:
    if len(dh_dict[key]) ==0:
        continue
    dh=np.concatenate([item.ravel() for item in dh_dict[key]])
    D2[key]={}
    D2[key]['med']=np.nanmedian(dh)
    D2[key]['count']=dh.size
    D2[key]['mad']=np.nanmedian(np.abs(dh-np.nanmedian(dh)))
    D2[key]['good_count']=np.nansum(np.abs(dh)<1).astype(np.float)
    if np.abs(D2[key]['med']) > 1 or D2[key]['mad'] > 5:
        print("HERE!")


keys=list(D2.keys())
meds=np.array([D2[key]['med'] for key in keys])
mads=np.array([D2[key]['mad'] for key in keys])
counts=np.array([D2[key]['count'] for key in keys])
good_counts=np.array([D2[key]['good_count'] for key in keys])
rgts=np.array([key[0] for key in keys])
mads[counts<100]=np.NaN
meds[counts<100]=np.NaN
good_counts[counts<100]=np.NaN


for ii in range(len(keys)):
    if ((np.abs(meds[ii])>1) or (mads[ii]>5)) and good_counts[ii]<10:
        print('cycle=%d, rgt=%d, med=%3.1f, mad=%3.1f, N=%d' % (keys[ii][1], keys[ii][0], meds[ii], mads[ii], counts[ii]))
        #print("%d,%d" % (int(keys[ii][0]), int(keys[ii][1])))

fig=plt.figure();
ax=fig.add_subplot(121)
ax.hist(np.log10(mads), 25)
ax.set_xlabel('log10(MAD, m)')
ax=fig.add_subplot(122)
ax.hist(meds, 25)
ax.set_xlabel('med, m')

plt.show()




 

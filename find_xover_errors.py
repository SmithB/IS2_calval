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

cycle=sys.argv[1]

ATL06_fields=['BP','LR','W','cycle_number','rgt','h_li','h_li_sigma','x','y','spot']

v,  delta,  bar = read_xovers(xover_base='/Volumes/ice2/ben/scf/AA_06/tiles', release='001', cycle=cycle)

# error analysis to pick out the worst RGTs:
DD={}
for col in [0, 1]:
    u_ct, D_ct = unique_by_rows(np.c_[np.round(v['rgt'][:, col]).astype(int), np.round(v['cycle_number'][:,col]).astype(int)], return_dict=True)
    for ct in u_ct:
        key=tuple(ct)
        if key not in DD:
            DD[key]={ki:np.zeros(2)+np.NaN for ki in ['mad','med','count']}
        dh=delta['h_li'][D_ct[key]]
        DD[key]['med'][col]=np.nanmedian(dh)
        DD[key]['mad'][col]=np.nanmedian(np.abs(dh))
        DD[key]['count'][col]=np.isfinite(dh).sum()

keys=list(DD.keys())
meds=np.array([DD[key]['med'] for key in keys])
mads=np.array([DD[key]['mad'] for key in keys])
counts=np.array([DD[key]['count'] for key in keys])
rgts=np.array([key[0] for key in keys])
mads[counts<10]=np.NaN
meds[counts<10]=np.NaN

for ii in range(len(keys)):
    if np.nanmax(np.abs(meds[ii,:])>5) or np.nanmax(mads[ii,:]>5):
        print('cycle=%d, rgt=%d, mad=%3.1f, abs(med)=%3.1f' % (keys[ii][1], keys[ii][0], np.nanmax(mads[ii,:]), np.nanmax(meds[ii,:])))
        print("%d,%d" % (int(keys[ii][1]), int(keys[ii][0])))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 09:02:21 2018

@author: ben
"""
import numpy as np
from read_DEM import read_DEM
from register_DEM import register_DEM
import matplotlib.pyplot as plt
import glob
import h5py
import scipy.interpolate as sI

files=glob.glob('/Volumes/ice2/ben/REMA_dems/8m/*.tif')

atmData=dict()
ATM_file='/Volumes/ice2/ben/ATM_WF/Bootleg/ATM_blockmedian_data_longline.h5'

with h5py.File(ATM_file,'r') as f:
    for field in ['x','y','h']:
        atmData[field]=np.array(f[field])

for file in files:
    delta_xy, sigma_xy, rVals, N_vals, biasSlope=register_DEM(file, atmData, \
       max_shift=500, delta_initial=50, delta_target=2., inATC=False, DOPLOT=False)
    print("%s, delta=%f %f, sigma=%f %f, N=%3.0f" % (file, delta_xy[0], delta_xy[1], sigma_xy[0], sigma_xy[0], N_vals))

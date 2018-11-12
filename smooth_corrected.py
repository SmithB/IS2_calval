# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:58:32 2018

@author: ben
"""

import scipy.ndimage.filters as sndf
import numpy as np

def smooth_corrected(I, sigma, mask=None):
    mask0=np.isfinite(I)
    if mask is not None:
        mask0[mask==0]=0
    Im=I.copy()
    Im[mask0==0]=0
    mf=sndf.gaussian_filter(mask0.astype(np.float64), sigma=sigma, mode='constant')
    Im=sndf.gaussian_filter(Im.astype(np.float64), sigma=sigma, mode='constant')
    Im[mf > 1.E-12]=Im[mf > 1.E-12]/mf[mf > 1.E-12]
    Im[mf <= 1.E-12]=np.NaN
    #Im[mask0==0]=np.NaN
    return Im

        
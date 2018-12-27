#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 15:35:13 2018

@author: ben
"""

from osgeo import gdal, gdalconst
import numpy as np
import matplotlib.pyplot as plt


def read_DEM(file, band_num=1, bounds=None, skip=1, asPoints=False, getProjection=False):
    """
        Read a raster from a DEM file
    """   
    ds=gdal.Open(file, gdalconst.GA_ReadOnly)
    if getProjection:
        proj=ds.GetProjection()
    band=ds.GetRasterBand(band_num)
    GT=ds.GetGeoTransform()
    nodataValue=band.GetNoDataValue()
    # ii and jj are the pixel center coordinates.  0,0 in GDAL is the upper-left
    # corner of the first pixel.
    ii=np.arange(0, band.XSize)+0.5
    jj=np.arange(0, band.YSize)-0.5
    x=GT[0]+GT[1]*ii
    y=GT[3]+GT[5]*jj
    if bounds is not None:
        cols = np.where(( x>=bounds[0][0] ) & ( x<= bounds[0][1] ))[0]
        rows = np.where(( y>=bounds[1][0] ) & ( y<= bounds[1][1] ))[0]
    else:
        rows=np.arange(band.YSize, dtype=int)
        cols=np.arange(band.XSize, dtype=int)
    
    z=band.ReadAsArray(int(cols[0]), int(rows[0]), int(cols[-1]-cols[0]+1), int(rows[-1]-rows[0]+1))
    ds=None
    
    if skip >1:
        z=z[::skip, ::skip]
        cols=cols[::skip]
        rows=rows[::skip]
    if nodataValue is not None and np.isfinite(nodataValue):
        bad = z==nodataValue
        z = np.float64(z)
        z[bad] = np.NaN
    else:
        z = np.float64(z)
    x=x[cols]
    y=y[rows]
    if asPoints:
        x,y=np.meshgrid(x, y)
        keep=np.isfinite(z.ravel())
        if getProjection:
            x.ravel()[keep], y.ravel()[keep], z.ravel()[keep], proj
        else:
            return x.ravel()[keep], y.ravel()[keep], z.ravel()[keep]
    else:
        if getProjection:
            return x, y[::-1], z[::-1, :], proj
        else:
            return x, y[::-1], z[::-1, :]
        
 
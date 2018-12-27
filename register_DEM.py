#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 09:02:21 2018

@author: ben
"""
import numpy as np
from read_DEM import read_DEM
import matplotlib.pyplot as plt
import osgeo
import scipy.interpolate as sI

def validate_xi(xy, xy0):
    # identify points that are inside an interpolation grid
    good=np.ones_like(xy[0], dtype=bool)
    for dim in [0,1]:
        good=good & (xy[dim] >= xy0[dim][0]) & (xy[dim] <= xy0[dim][-1])
    return good


def evaluate_shift(dxy, basis_vectors, Dsub, gI, inATC=False, return_shifted=False):
    """ 
        evaluate the misfit between altimetry measurements and a DEM 
        for one shift vector
    """
    # returns: R, N, biasSlope, this_delta, dh 
    # define the offset as a function of the basis vectors
    this_delta = dxy[0]*basis_vectors[0] + dxy[1]*basis_vectors[1]
    
    hi=np.zeros_like(Dsub['x'])+np.NaN
    good=validate_xi((Dsub['y']+this_delta[1], Dsub['x']+this_delta[0]), gI.grid)
    # interpolate the DEM values at the offset data points
    hi[good]=gI((Dsub['y'][good]+this_delta[1], Dsub['x'][good]+this_delta[0]))
    # calculate the difference between the data values and the interpolated values
    dh=Dsub['h']-hi
    ii=np.isfinite(dh)
    if inATC:
        # Solve for the residual slope in the along-track direction
        G=np.c_[np.ones(ii.sum), \
                (Dsub['y'][ii]-Dsub['y'][ii].mean())*basis_vectors[0][1] + \
                (Dsub['x'][ii]-Dsub['x'][ii].mean())*basis_vectors[0][0]]
        # degrees of freedom are Ndata minus two for the fit, one for dx and one for dy
        nDF=G.size[0]-4.
    else:
        # solve for the best-fitting plane for the residuals
        G=np.c_[np.ones(ii.sum()), \
                (Dsub['x'][ii]-Dsub['x'][ii].mean()).ravel()/1000.,\
                (Dsub['y'][ii]-Dsub['y'][ii].mean()).ravel()/1000.]
        # degrees of freedom are Ndata minus three for the fit, one for dx and one for dy
        nDF=G.shape[0]-5.
    # solve the normal equations for the model      
    try:          
        biasSlope=np.linalg.solve(G.transpose().dot(G), G.transpose().dot(dh[ii]))  
    except np.linalg.LinAlgError:
        biasSlope=np.array([dh[ii].mean(), 0.])
    # calculate the mean-squared residual
    R=np.sum((dh[ii]-G.dot(biasSlope))**2)/nDF
    # save the data count
    N=np.sum(ii)
    if return_shifted:
        return R, N, biasSlope, this_delta, ii, dh[ii]-G.dot(biasSlope)
    else:
        return R, N, biasSlope


def register_DEM(DEMfile, pointData, max_shift=500, delta_initial=50, delta_target=2., inATC=False, DOPLOT=False):
    delta_xy=np.array([np.NaN, np.NaN])
    sigma_xy=np.array([np.NaN, np.NaN])
    N_best=np.NaN
    R_best=np.NaN

    # read the DEM data
    DEM=dict()
    DEM['x'], DEM['y'], DEM['z'], projSys=read_DEM(DEMfile, getProjection=True)
    demRes=DEM['x'][1]-DEM['x'][0]
    # calculate the projection from latlon to the DEM CS
    llRef = osgeo.osr.SpatialReference()
    llRef.ImportFromEPSG(4326)
    demRef = osgeo.osr.SpatialReference()
    demRef.ImportFromWkt(projSys)
    xform = osgeo.osr.CoordinateTransformation(llRef,demRef)
    # project the point data into the DEM coordinates
    if 'x' not in pointData:
        xy=np.array(xform.TransformPoints( np.c_[pointData['longitude'], pointData['latitude'], np.zeros_like(pointData['longitude'])]))[:,0:2]
        pointData['x']=xy[:,0]
        pointData['y']=xy[:,1] 

    # calculate the basis vectors for the offsets
    basis_vectors=list()
    if inATC:
        # if the in_ATC keyword is set, the first basis vector is parallel to the difference between the first and last points
        basis_vectors.append( np.array([pointData['x'][-1]-pointData['x'][0], pointData['y'][-1]-pointData['y'][0]]) )
        basis_vectors[0] = basis_vectors[0] / np.sqrt(np.sum(basis_vectors[0]**2))
        # the second vector is perpendicular to the first
        basis_vectors.append(np.array([-basis_vectors[0][1], basis_vectors[0][0]]))
    else:
        # if inATC is false, just use x and y
        basis_vectors.append( np.array([1., 0.]) )
        basis_vectors.append( np.array([0., 1.]) )
    # set up the geo interpolation object
    gI=sI.RegularGridInterpolator((DEM['y'], DEM['x']), DEM['z'], fill_value=np.NaN)
       
    # select the subset of the data that are within the DEM box
    valid=validate_xi((pointData['y'], pointData['x']), gI.grid)
    Dsub=dict()
    for field in pointData.keys():
        Dsub[field]=pointData[field][valid]
    # next select the subset of the subset that has valid DEM values    
    valid=np.isfinite(gI((Dsub['y'], Dsub['x'])))
    for field in Dsub.keys():
        Dsub[field]=Dsub[field][valid]
    
    if Dsub['x'].size < 10 or (np.max(Dsub['x'])-np.min(Dsub['x'])) < 100*demRes or (np.max(Dsub['y'])-np.min(Dsub['y'])) < 100*demRes :
        print("not enough data to span the offsets")
        if inATC:
            return delta_xy, sigma_xy, R_best, N_best, np.zeros(2)+np.NaN, basis_vectors
        else:
            return delta_xy, sigma_xy, R_best, N_best, np.zeros(3)+np.NaN
      
    # Define the offsets that will be used to bracket each potential minimum
    dxsub, dysub=np.meshgrid(np.array([-1., 0., 1.]), np.array([-1., 0., 1.]))

    # delta is the spacing between test offset values.  It will be reduced iteratively
    delta=delta_initial
    # define a grid of search values that span the initial range (-max_shift to +max_shift)
    [x_offsets, y_offsets]=np.meshgrid(np.arange(-max_shift, max_shift+delta, delta), \
        np.arange(-max_shift, max_shift+delta, delta))
    # store the offset values we have searched at the residuals and counts at each in a set of dictionaries
    R=dict()
    N=dict()
    biasSlope=dict()
        
    # loop until delta has been reduced to delta_target (the /2 means that the last
    # iteration has delta <= delta_target)
    while delta > delta_target/2:
        # Search all offset values defined by the previous iteration
        for dxy in zip(x_offsets.ravel(), y_offsets.ravel()):
            # skip this value if it is already in the R dictionary
            if dxy in R: 
                continue          
            R[dxy], N[dxy], biasSlope[dxy] = evaluate_shift(dxy, basis_vectors, Dsub, gI, inATC)
        # identify the offsets we've searched
        searched=list(R)
        # calculate the residual values for the keys
        rVals=[R[ii] for ii in searched]
        # find the minimum residual
        off_best=searched[np.argmin(rVals)]
        # define the offsets for the next iteration
        x_offsets=list()
        y_offsets=list()
        # check if the eight points surrounding the current point (+- delta in each direction)
        # have been evaluated.  If not, search them.  If all eight have been searched, 
        # refine delta by a factor of two
        while len(x_offsets) == 0:
            for dxi, dyi in zip(dxsub.ravel(), dysub.ravel()):
                this_offset=(off_best[0]+dxi*delta, off_best[1]+dyi*delta)
                if this_offset not in R:
                    # if this offset has not been searched, add it to the list
                    x_offsets.append(this_offset[0])
                    y_offsets.append(this_offset[1])                    
            # if all offsets have been searched, loop to a smaller delta
            if len(x_offsets)==0:
                delta /= 2
        x_offsets=np.array(x_offsets)
        y_offsets=np.array(y_offsets)
    # collect outputs
    N_vals=np.array([N[ii] for ii in searched])
    xyOff=np.c_[searched]
    # find the best offset
    iBest=np.argmin(rVals)
    delta_xy=np.array(off_best)
    # identify the offsets that are not significantly different from the minimum
    # assuming the errors are independent and normally distributed 
    inEB=np.where(rVals < rVals[iBest]*(1+2/N_vals[iBest]))[0]
    # sigma_xy is half the range of offsets that are not significantly different
    # from the minimum
    sigma_xy=np.array([(np.max(xyOff[inEB, 0])-np.min(xyOff[inEB, 0]))/2.,\
                       (np.max(xyOff[inEB, 1])-np.min(xyOff[inEB, 1]))/2.])  
  
    if DOPLOT:
        plt.figure()
        plt.subplot(311)
        plt.scatter(xyOff[:,0], xyOff[:,1], c=np.array(rVals))
        plt.plot(delta_xy[0], delta_xy[1],'ko',linewidth=2)
        plt.axis('equal')
        
        R, N, tempBiasSlope, this_delta, ii, dh = evaluate_shift( (0. ,0.), basis_vectors, Dsub, gI, inATC, return_shifted=True)
        vrange=np.nanmean(dh)+np.array([-2*np.sqrt(R), 2*np.sqrt(R)])
        plt.subplot(312)
        plt.scatter(Dsub['x'][ii], Dsub['y'][ii], c=dh, linewidth=0, vmin=vrange[0], vmax=vrange[1])
        plt.colorbar()
        plt.axis('equal')
        plt.subplot(313)
        R, N, tempBiasSlope, this_delta, ii, dh = evaluate_shift( delta_xy, basis_vectors, Dsub, gI, inATC, return_shifted=True)
        plt.scatter(Dsub['x'][ii], Dsub['y'][ii], c=dh, linewidth=0, vmin=vrange[0], vmax=vrange[1]) 
        if inATC:
            # plot the basis vectors
            W=(np.nanmax(Dsub['x'])-np.nanmin(Dsub['y']))
            plt.arrow((Dsub['x'][0], Dsub['y'][0]), W/4*basis_vectors[0], color='k')
            plt.arrow((Dsub['x'][0], Dsub['y'][0]), W/4*basis_vectors[1], color='r') 
        plt.colorbar()
        plt.axis('equal')

    if inATC:
        # return the basis vectors if we're working in along-track coordinates
        return delta_xy, sigma_xy, rVals[off_best], N_vals[off_best], biasSlope[off_best], basis_vectors
    else:
        # otherwise, don't.
        return delta_xy, sigma_xy, rVals[iBest], N_vals[iBest], biasSlope[off_best]
  
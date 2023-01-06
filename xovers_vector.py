#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:51:17 2019

@author: ben
"""
import numpy as np
import pointCollection as pc

def x_point(A, B):
    """ 
    Find crossing points between segments in complex matrices
    """
    
    # assumes that A and B contain nxm matrices of complex coordinates,
    # where each row is a path 
    dA=A[:,-1]-A[:,0]
    dB=B[:,-1]-B[:,0]
    det = -(dA*(dB.conj())).imag
    dAB0 = A[:,0]-B[:,0]
    lA = (dAB0*(dB.conj())).imag/det
    lB = (dAB0*(dA.conj())).imag/det
    #print('x_point:'+str([lA, lB]))
    status = ( lA>0 ) & ( lA < 1) & ( lB > 0) & ( lB <1 )
    
    return lA, lB, A[:,0]+lA*dA, status

def sub_path(Dc, rows, cols):
    """
        Extract the segments starting in columns cols[0] and cols[1]
        on specified rows from Dc
    """
    p_subs=[np.zeros((len(rows), 2), dtype=complex) for ii in [0, 1]]
    for Di, col, p_sub in zip(Dc, cols, p_subs):
        for col1 in [0, 1]:
            ind=np.ravel_multi_index((rows, col+col1), Di.shape)
            p_sub[:, col1]=Di.ravel()[ind]
    return p_subs

def guess_first_xover(Dc):

    L0, L1, xy0, status = x_point(Dc[0][:,[0, -1]], Dc[1][:,[0, -1]])

    # search based on along-track distance to find the first segment of the crossing point:
    cols=[[], []]
    for ii, Di, Li in zip([0, 1], Dc, [L0, L1]):
        t_hat = Di[:,-1]-Di[:,0]
        L=np.abs(t_hat)
        t_hat /= L**2

        x_atc = np.tile(t_hat.conj()[:,None], [1,Di.shape[1]])*(Di-Di[:,0][:, None])
        cols[ii] = np.argmin(Li[:, None]>x_atc, axis=1)-1
    return cols, status


def cross_paths(Dc):
    """
    Find intersections between paths given in Dc.  
    
    Inputs:
        Dc : iterable of two complex matrices, with one 
                path on each row.
    outputs:
        list : indices into the columns of Dc.
        list : Fractional offsets between the two values in each row
        boolean array: status of each crossover (true indicates that paths cross)
        count: number of iterations
    """
    count=0
             
    # find the crossing point between the endpoints of the paths
    count += 1 
    cols, status=guess_first_xover(Dc)
 
    # we will examine only those rows that had a successful crossing the first time
    rows = np.flatnonzero(status)
    
    L0, L1 = [np.zeros((Dc[0].shape[0]))+np.NaN for ii in [0, 1]]
    i0, i1 = [np.zeros((Dc[0].shape[0]), dtype=int) for ii in [0, 1]]
    
    cols = [col[rows] for col in cols]
    count=0
    while count < 36:
        count += 1
        sub_paths = sub_path(Dc, rows, cols)
        
        L0sub, L1sub, xy0sub, status_sub = x_point(*sub_paths)
        
        rr=rows[status_sub]
                     
        #write out successes
        L0[rr] = L0sub[status_sub]
        L1[rr] = L1sub[status_sub]
        i0[rr] = cols[0][status_sub]
        i1[rr] = cols[1][status_sub]
        
        # shift the failures
        cols[0][L0sub<0] -= 1
        cols[0][L0sub>1] += 1
        cols[1][L1sub<0] -= 1
        cols[1][L1sub>1] += 1
  
                
        # continue on those paths for which the search isn't out of bounds, and
        # for which we haven't found a crossover
        cc=np.c_[cols]
        search_more = np.all((cc >0) & (cc < Dc[0].shape[1]-2), axis=0) & (status_sub==0)
        if ~np.any(search_more):
            break
        for ii in [0, 1]:
            cols[ii] = cols[ii][search_more]
        rows = rows[search_more]
    return [i0, i1], [L0, L1], np.isfinite(L0), count
    
    
   
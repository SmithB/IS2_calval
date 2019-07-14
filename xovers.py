#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:51:17 2019

@author: ben
"""
import numpy as np
import matplotlib.pyplot as plt
from PointDatabase import point_data
def x_point(A, B):

    dA=A[-1]-A[0]
    dB=B[-1]-B[0]
    det=-np.imag(dA*(dB.conjugate()))
    dAB0=A[0]-B[0]
    lA=np.imag(dAB0*(dB.conjugate()))/det
    lB=np.imag(dAB0*(dA.conjugate()))/det

    if (lA <0 ) or (lA >1) or (lB<0) or (lB >1):
        return None, None, None

    return lA, lB, A[0]+lA*dA

def reduce_interval(t, f, ind, mode='largest'):
    ti=np.array([t[ind][0], t[ind][1]])
    tf=t[ind[0]]+(ti[1]-ti[0])*f
    if mode == 'both':
        ind=[np.maximum(0, first_true(t>tf)-1),
            np.minimum(len(t)-1, first_true(t>tf))]
        return ind
    if tf-ti[0] > ti[1]-tf:
        # subdivide the first interval
        ind[0]=np.minimum(ind[1]-1, np.where(t>(tf+ti[0])/2)[0][0])
    else:
        ind[1]=np.maximum(ind[0]+1, np.where(t<(tf+ti[1])/2)[0][-1] )

def cross_paths(D):
    """
    Function that finds an intersection between two paths given in D.  The paths
    must be structures, with fields .x .y, and .time.  Each must be dense (i.e. all elements
    must be present, no gaps allowed.
    """
    Dc=list()
    times=list()
    inds=list()
    for ii, Di in enumerate(D):
        Dc.append(Di.x+1j*Di.y)
        times.append(Di.time)
        inds.append([0, len(Di.x)-1])
    while (inds[0][-1]-inds[0][0] >1) and (inds[1][-1]-inds[1][0] >1):
        plt.figure();
        for ii in [0, 1]:
            plt.plot(D[ii].x[inds[ii][0]:inds[ii][1]+1], D[ii].y[inds[ii][0]:inds[ii][1]+1])
            plt.plot(D[ii].x[inds[ii]], D[ii].y[inds[ii]])
        F = x_point(Dc[0][inds[0]], Dc[1][inds[1]])
        if F[0] is None or F[1] is None:
            return None, None
        # try to reduce the interval to the segments adjacent to the estimated location from cross_point
        iTemp=[[], []]
        for ii in [0, 1]:
            iTemp[ii]=reduce_interval(times[ii], F[ii], inds[ii].copy(), mode='both')
        F1=x_point(Dc[0][iTemp[0]], Dc[1][iTemp[1]])
        if F1[0] is not None and F1[1] is not None:
            return iTemp, F1
        # if reducing  to the single segment did not work, split the largest inteval, and continue
        for ii in [0, 1]:
            reduce_interval(times[ii], F[ii], inds[ii])
    return inds, F


import matplotlib.pyplot as plt
x0=np.arange(0, 13, 2)
y0=0.1*(x0*2)**2-2
x1=np.arange(0.5, 5.2, 1.)
y1=-(x1**2)+x1+5

plt.figure()
plt.plot(x0, y0)
plt.plot(x1, y1)

D=[point_data().from_dict({'x':x0, 'y':y0, 'time':np.arange(len(x0))}),
   point_data().from_dict({'x':x1, 'y':y1, 'time':np.arange(len(x1))})]


ii, f=cross_paths(D)

print(x0[ii[0]:ii[0]+2].dot([1-f[0], f[0]]) - x1[ii[1]:ii[1]+2].dot([1-f[1], f[1]]))
print(y0[ii[0]:ii[0]+2].dot([1-f[0], f[0]]) - y1[ii[1]:ii[1]+2].dot([1-f[1], f[1]]))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:51:17 2019

@author: ben
"""
import numpy as np
import matplotlib.pyplot as plt
def crossPoint(A, B):
    A0, B0=np.meshgrid(A[0:-1], B[0:-1])
    A1, B1=np.meshgrid(A[1:], B[1:])
    dA=A1-A0
    dB=B1-B0
    det=-np.imag(dA*(dB.conjugate()))
    dAB0=A0-B0
    lA=np.imag(dAB0*(dB.conjugate()))/det
    lB=np.imag(dAB0*(dA.conjugate()))/det

    goodXOs=(lA>=0) & (lA<=1) & (lB>=0) & (lB<=1)
    iB, iA=np.where(goodXOs)

    if len(iA)==0 or len(iB)==0:
        return None, None, None, None

    lA=lA[iB, iA]
    lB=lB[iB, iA]
    Ai=A[iA]*(1-lA)+A[iA+1]*lA
    Bi=B[iB]*(1-lB)+B[iB+1]*lB

    plt.plot(np.real(A), np.imag(A),'ro')
    plt.plot(np.real(B), np.imag(B),'bo')

    plt.plot(np.real(A[iA]), np.imag(A[iA]),'rx')
    plt.plot(np.real(B[iB]), np.imag(B[iB]),'bx')
    plt.plot(np.real(Ai), np.imag(Ai),'r*')
    plt.plot(np.real(Bi), np.imag(Bi),'b*')

    return iA[0], iB[0], lA, lB


def crossPaths(D):
    """
    Function that finds an intersection between two paths given in D.  The paths
    may be dictionaries with entries 'x' and 'y', or structures, with fields .x
    and .y.  Each must be dense (i.e. all elements must be present, no gaps allowed.
    """


    Dc=list()
    time=list()
    intervals=list()
    tDeltas=np.zeros(2)
    dt=np.zeros(2)
    for ii, Di in enumerate(D):
        if hasattr(Di,'x'):
            Dc.append(Di.x+1j*Di.y)
        else:
            Dc.append(Di['x']+1j*Di['y'])
        if hasattr(Di,'time'):
            time.append(Di.time)
        else:
            time.append(Di['time'])
        intervals.append([0, Dc[-1].size-1])
        tDeltas[ii]=time[ii][-1]-time[ii][0]
        dt[ii]=time[ii][1]-time[ii][0]
    t_est=np.zeros(2)
    while (intervals[0][-1]-intervals[0][0] >1) and (intervals[1][-1]-intervals[1][0] >1):
        for interval in intervals:
            if interval[-1]-interval[0]==1:
                ind.append(interval)
            else:
                ind.append([interval[0], int(np.floor((interval[1]+interval[0])/2)), interval[1]])
        l0=crossPoint(Dc[0][intervals[0]], Dc[1][intervals[1]])
        if i0 is None or i1 is None:
            return None, None
        for ii in [0, 1]:
            t_est[ii]=time[ii][intervals[ii]].dot([1-l0[ii], l0[ii]])
            intervals[ii]=[]

        intervals[0]=[np.maximum([0, i0_est-deltas[0]]), np.minimum[Dc[0].size, i0_est+deltas[0]]]

        deltas=np.ceil(deltas/2)
    return intervals[0][0], intervals[1][0], l1, l2


import matplotlib.pyplot as plt
x0=np.arange(-11, 13, 2)
y0=0.1*(x0*2)**2-2
x1=np.arange(0.5, 5.2, 1.)
y1=-(x1**2)+x1+5

plt.figure()
plt.plot(x0, y0)
plt.plot(x1, y1)

i0, i1, l0, l1=crossPaths([{'x':x0, 'y':y0}, {'x':x1, 'y':y1}])

print(x0[i0:i0+2].dot([1-l0, l0]) - x1[i1:i1+2].dot([1-l1, l1]))
print(y0[i0:i0+2].dot([1-l0, l0]) - y1[i1:i1+2].dot([1-l1, l1]))


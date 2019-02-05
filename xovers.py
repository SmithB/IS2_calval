#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 23:51:17 2019

@author: ben
"""
import numpy as np

def crossPoint(A, B):
    A0, B0=np.meshgrid(A[0:-1], B[0:-1])
    A1, B1=np.meshgrid(A[0:-1], B[0:-1])
    dA=A1-A0
    dB=B1-B0
    det=-np.imag(dA.dot(dB.conjugate()))
    dAD0=A0-B0
    lA=np.imag(dAD0.dot(dB.conjugate()))/det
    lB=np.imag(dAD0.dot(dA.conjugate()))/det

    goodXOs=(lA>0) & (lA<1) & (lB>0) & (lB<1)
    iA, iB=np.where(goodXOs)

    if len(iA)==0 or len(iB)==0:
        return None, None, None, None
    lA=lA[iA, iB]
    lB=lB[iA, iB]

    return iA, iB, lA, lB


def crossPaths(D):
    Dc=list()
    intervals=list()
    for Di in D:
        if hasattr(Di,'x'):
            Dc.append(Di.x+1j*Di.y)
        else:
            Dc.append(Di['x']+1j*Di['y'])
        intervals.append([0, Dc[-1].size-1])

    while np.max([interval[-1]-interval[0] for interval in intervals])>1:
        ind=list()
        for interval in intervals:
            if interval[-1]-interval[0]==0:
                ind.append(interval)
            else:
                ind.append([interval[0], int(np.floor((interval[1]+interval[0])/2)), interval[1]])
        i1, i2, l1, l2=crossPoint(D[0][ind[0]], D[1][ind[1]])
        if i1 is None or i2 is None:
            return None, None, None, None
        intervals[1]=ind[i1:i1+1]
        intervals[2]=ind[i2:i2+1]

    return i1, i2, l1, l2




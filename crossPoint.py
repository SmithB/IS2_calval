#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:58:42 2019

@author: ben
"""

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


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
DOPLOT=True

def regular_grid_interp_mtx(x0, xi, delta=None):
    # fast linear interpolation matrix script, creates a sparse matrix that, when dotted with a vector of nodal values, returns the interpolated values at the specified data points
    if delta is None:
        delta=x0[1]-x0[0]
    inBds=np.where((xi >= x0[0]) & (xi < x0[-1]))[0]
    ii=(xi[inBds]-x0[0])/delta
    di=ii-np.floor(ii)
    M=sps.coo_matrix(np.c_[1-di, di], (np.c_[inBds, inBds], np.c_[np.floor(ii), np.floor(ii)+1].astype(int)), shape=[xi.size, x0.size]).tocsr()
    return M, inBds

def gaussian(x, ctr, sigma):
    # return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def fit_shifted(delta_t, sigma, D0, D1, M, key_in, key_top):     
    R=np.zeros_like(delta_t)
    G=np.zeros((D1['t'].size, 2))

    if DOPLOT:
        plt.clf()
        plt.plot(D1.t, D1.p,'k')
    for ii,t_val in enumerate(delta_t):
        this_key=tuple(key_in+[t_val])
        if this_key in M:
            R[ii]=M[this_key]['best']['R']
        else:
            if this_key not in D0:
                D0[this_key]=dict()
            if ['interp_mat'] not in D0[this_key]:
                D0[this_key]['interp_mat'], D0[this_key]['inBds']=regular_grid_interp_mtx(D0[key_in]['t']-t_val, D1['t'])
            D0[this_key]['p']=D0[this_key]['interp_mat'].dot[D0[this_key]['p']]
            G[:,1]=D0[this_key]['p']
            m=np.inv(G.transpose().dot(G)).dot(G.dot(D1['p']))
            R[ii]=np.sum((D1['p']-G.dot(m))**2)
            M[this_key]={'K0':key_top,'R':R[ii],'A':m[1],'B':m[0],'dt':delta_t,'sigma':sigma}  
            if DOPLOT:
                plt.plot(D1.t, G.dot(m))
    if DOPLOT:
        plt.figure(plt.gcf())
        plt.legend(R)
        plt.pause(2)
    iR=np.argsort(R)
    this_key=key_in+[delta_t[iR[0]]]
    M[tuple(key_in)]['best']={'key':tuple(this_key),'R':R[iR[0]]}
    this_key=key_in+[delta_t[iR[1]]]
    M[tuple(key_in)]['second_best']={'key':tuple(this_key),'R':R[iR[1]]}
    return 

def fit_broadened(sigmas, delta_t, D0, D1, M, key_in, key_top):     
    R=np.zeros_like(sigmas)
    for ii, sigma in enumerate(sigmas):         
        this_key=key_in+[sigma]
        if tuple(this_key) in M:
            R[ii]=M[this_key]
            continue
        if tuple(this_key) not in D0:
            # if we haven't already broadened the WF to sigma, try it now:
            dt=D0.t[1]-D0.t[0]
            nK=3*np.ceil(sigma/dt)
            tK=np.arange(-nK, nK+1)*dt
            K=gaussian(tK, 0, sigma)
            D0[tuple(this_key)]={'t':D0[key_top]['t'], 'p':D0[key_in]['interp_mtx'].dot(np.convolve(D0[key_top]['p'], K,'same'))}         
        
        if ii>0 and R[ii]>R[ii-1]:
            break
    iR=np.argsort(R[0:ii])
    this_key=key_in+[sigmas[iR[0]]]
    M[tuple(key_in)]['best']={'key':tuple(this_key),'R':R[iR[0]]}
    this_key=key_in+[delta_t[iR[1]]]
    M[tuple(key_in)]['second_best']={'key':tuple(this_key),'R':R[iR[1]]}
    return 
    
    
def fit_library(D1s, D0s, sigmas, delta_ts):
    D0=dict()
    for D1 in D1s:
        M=dict()
        for kk in list(D0s):
            if kk not in D0:
                D0[kk]=D0s[kk].copy()
            fit_broadened(sigmas, delta_ts, D0, D1, M, kk, kk)
        best_model=M
        while 'best' in best_model:
            best_model=best_model['best']
        D1.update(best_model)
    return 
    
def test():
    t=np.arange(0, 10, 0.01)
    tg=np.arange(-15., 15.)
    K=gaussian(tg, 0, 3)
    p=np.exp(-(t-5)/.25)
    p[t<5]=0
    p=np.convolve(p, K, 'same')
    D1s=list({'t':t, 'p':p+.25})
    D0s={[1]:{'t':tg,'p':gaussian(tg, 0, 1)}, [2]:{'t':tg,'p':gaussian(tg,0, 2)}}
    delta_ts=np.arange(-6, 6, 0.25)
    sigmas=np.arange(0, 4, 0.25)
    fit_library(D1s, D0s, sigmas, delta_ts)
 
    print(D1s)
if __name__=="__main__":
    test()
    
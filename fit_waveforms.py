# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import bisect
DOPLOT=False

class listDict(dict):
    def __setitem__(self, key, value):
        if isinstance(key, list):
            dict.__setitem__(self, tuple(key), value)
        else:
            dict.__setitem__(self, key, value)
    def __getitem__(self, key):
        if isinstance(key, list):
            return dict.__getitem__(self, tuple(key))
        else:
            return dict.__getitem__(self, key)
    def __contains__(self, key):
        if isinstance(key, list):
            return dict.__contains__(self, tuple(key))
        else:
            return dict.__contains__(self, key)
        
        

def regular_grid_interp_mtx(x0, xi, delta=None):
    # fast linear interpolation matrix script, creates a sparse matrix that, when dotted with a vector of nodal values, returns the interpolated values at the specified data points
    if delta is None:
        delta=x0[1]-x0[0]
    inBds=np.where((xi >= x0[0]) & (xi < x0[-1]))[0]
    ii=(xi[inBds]-x0[0])/delta
    di=ii-np.floor(ii)
    
    M=sps.coo_matrix((np.c_[1-di, di].ravel(), (np.c_[inBds, inBds].ravel(), np.c_[np.floor(ii), np.floor(ii)+1].ravel().astype(int))), shape=(xi.size, x0.size)).tocsr()
    return M, inBds

def gaussian(x, ctr, sigma):
    # return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def lin_fit_misfit(x, y, G=None, m=None, return_data_est=False):
    if G is None:
        G=np.ones((x.size, 2))
    G[:,0]=x.ravel()
    good=np.isfinite(G[:,0]) & np.isfinite(y.ravel())
    G1=G[good,:]
    if m is None:
        m=np.linalg.inv(G1.transpose().dot(G1)).dot(G1.transpose().dot(y[good]))
    R=np.sum((y[good]-G1.dot(m))**2)
    if return_data_est:
        y_est=np.zeros_like(y)+np.Nan
        y_est[good]=G1.dot(m)
        return R, m, y_est
    else:
        return R, m, y_est

def shifted_misfit(delta_t, sigma, WF, deck, M, key_in, key_top, G=None, return_data_est=False):
    if G is None:
        G=np.ones((WF['t'].size, 2))
    this_key=key_in+[delta_t]
    if this_key in M:
        R=M[this_key]['R']     
    else:
        M[this_key]=listDict()
        if this_key not in deck:
            deck[this_key]=listDict()            
            interp_mat,inBds=regular_grid_interp_mtx(deck[key_in]['t']-delta_t, WF['t'])
            deck[this_key]['p']=interp_mat.dot(deck[key_in]['p'])
            deck[this_key]['t']=deck[key_in]['t']
        R, m, wf_est=lin_fit_misfit(deck[this_key]['p'], WF['p'], G=G)
        M[this_key]={'K0':key_top, 'R':R, 'A':m[0], 'B':m[1], 'dt':delta_t, 'sigma':sigma}  
        
        if DOPLOT:
            plt.plot(WF['t'], G.dot(m))
        if return_data_est:
            return R, G.dot(m)
        else:
            return R
        
def fit_shifted(delta_t_list, sigma, deck, WF, M, key_in, key_top, t_tol):     
    R_dict=dict()
    G=np.ones((WF['t'].size, 2))

    if DOPLOT:
        plt.clf()
        plt.plot(WF['t'], WF['p'],'k')
    # first search the (coarse) input values of delta_t
    delta_t=delta_t_list.copy()
    delta_t_searched=list
    while (len(delta_t_searched)==0) or  (np.diff(np.array(delta_t_searched)).min > t_tol) :       
        for t_val in delta_t:
            R_dict[t_val]=shifted_misfit(t_val, sigma, WF, deck, M, key_in, key_top, G=G)
            bisect.lsort_left(delta_t_searched, t_val)
        # make a list of R_vals searched
        R_vals=[R_dict[t_val] for t_val in delta_t_searched]
        # sort the R_vals
        iR=np.argsort(R_vals)
        # The next search value is the golden-rule value (0.7 of the way between 
        #  the times for the best and second best residuals)
        delta_t=list((0.7*delta_t_searched[iR[0]]+0.3*delta_t_searched[iR[1]]))
 
    this_key=key_in+[delta_t_searched[iR[0]]]
    M[key_in]['best']={'key':this_key,'R':R_vals[iR[0]]}
    return R_vals[iR[0]]

def fit_broadened(sigmas, delta_ts, deck, D1, M, key_in, key_top):     
    R=np.zeros_like(sigmas)
    for ii, sigma in enumerate(sigmas):         
        this_key=key_in+[sigma]
        if this_key in M:
            R[ii]=M[this_key]
            continue
        else:
            M[this_key]=listDict()
            if this_key not in deck:
                # if we haven't already broadened the WF to sigma, try it now:
                if sigma==0:
                    deck[this_key]={'t':deck[key_top]['t'], 'p':deck[key_top]['p']}
                else:    
                    dt=np.diff(deck[key_top]['t'][0:2])
                    nK=3*np.ceil(sigma/dt)
                    tK=np.arange(-nK, nK+1)*dt
                    K=gaussian(tK, 0, sigma)
                    deck[this_key]={'t':deck[key_top]['t'], 'p':np.convolve(deck[key_top]['p'], K,'same')}         
            R[ii]=fit_shifted(delta_ts, sigma, deck, D1, M, this_key, key_top)            
        if ii>0 and R[ii]>R[ii-1]:
            break
    iR=np.argsort(R[0:ii+1])
    this_key=key_in+[sigmas[iR[0]]]
    M[key_in]['best']={'key':this_key,'R':R[iR[0]]}
    this_key=key_in+[delta_ts[iR[1]]]
    M[key_in]['second_best']={'key':this_key,'R':R[iR[1]]}
    return R[iR[0]]
    
    
def fit_library(D1s, deck_in, sigmas, delta_ts):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms
    
    """
    # make an empty container where we will keep waveforms we've tried already
    deck=listDict()
    # loop over input waveforms
    for D1 in D1s:
        # set up a matching dictionary (contains keys of waveforms and their misfits)
        M=listDict()
        # loop over the library of templates
        keys=list(deck_in)
        R=np.zeros(len(keys))
        for ii, kk in enumerate(keys):
            # check if we've searched this template before, otherwise copy it into
            # the library of checked templates
            if kk not in deck:
                deck[[kk]]=deck_in[kk].copy()
            if kk not in M:
                M[[kk]]=listDict()
            # find the best misfit between this template and the waveform
            R[ii]=fit_broadened(sigmas, delta_ts, D0, D1, M, [kk], [kk])
        # recursively search the M dict for the best match
        best_model=M
        while 'best' in best_model:
            best_model=best_model['best']
        # write out the best model information to the input waveform
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
    D0s={(1):{'t':tg,'p':gaussian(tg, 0, 1)}, (2):{'t':tg,'p':gaussian(tg,0, 2)}}
    delta_ts=np.arange(-6, 6, 0.25)
    sigmas=np.arange(0, 4, 0.25)
    fit_library(D1s, D0s, sigmas, delta_ts)
 
    print(D1s)
if __name__=="__main__":
    test()
    
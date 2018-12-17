# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
import matplotlib.pyplot as plt
import bisect
from waveform import waveform
DOPLOT=False


class listDict(dict):
    """
    Subclass of a dictionary, that can take lists as keys.  Any list key is converted
    to a tuple, so it will be returned by the keys() method as a tuple.
    """
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

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def lin_fit_misfit(x, y, G=None, m=None):
    """
    Calculate the best-fitting background + scaled waveform model, return its
    misfit 
    """
    if G is None:
        G=np.ones((x.size, 2))
    G[:,0]=x.ravel()
    good=np.isfinite(G[:,0]) & np.isfinite(y.ravel())
    # need at least three good values to calculate a misfit
    if good.sum() < 3:
        m=np.zeros(2)
        R=np.sqrt(np.sum(y**2)/(y.size-2))
        return R, m
    G1=G[good,:]
    try:
        m=np.linalg.solve(G1.transpose().dot(G1), G1.transpose().dot(y[good]))
        R=np.sqrt(np.sum((y[good]-G1.dot(m))**2.)/(good.sum()-2))
        #R=np.sqrt(m_all[1][0])
    except np.linalg.LinAlgError:
        m=np.zeros(2)
        R=np.sqrt(np.sum(y**2.)/(y.size-2))
    return R, m

def wf_misfit(delta_t, sigma, WF, catalog, M, key_top,  G=None, return_data_est=False):
    """    
        Find the misfit between a scaled and shifted template and a waveform
    """
    if G is None:
        G=np.ones((WF.p.size, 2))
    this_key=key_top+[sigma]+[delta_t]
    if (this_key in M) and (return_data_est is False):
        return M[this_key]['R']     
    else:
        # check if the broadened but unshifted version of this key is in the catalog
        broadened_key=key_top+[sigma]+[0]
        if broadened_key in catalog:
            broadened_wf=catalog[broadened_key].p
        else:
            # make a broadened version of the catalog WF
            if sigma==0:
                 broadened_wf = catalog[key_top].p
            else:
                nK=3*np.ceil(sigma/WF.dt)
                tK=np.arange(-nK, nK+1)*WF.dt
                K=gaussian(tK, 0, sigma)
                broadened_wf=np.convolve(catalog[key_top].p, K,'same')
            catalog[broadened_key]=waveform(catalog[key_top].t, broadened_wf, t0=catalog[key_top].t0, tc=catalog[key_top].tc)
        if this_key not in catalog:
            M[this_key]=listDict()
            catalog[this_key]=catalog[broadened_key].copy()          
            catalog[this_key].p=np.interp(WF.t-WF.tc, catalog[key_top].t-catalog[key_top].tc+delta_t, broadened_wf)
            
        R, m = lin_fit_misfit(catalog[this_key].p, WF.p, G=G)
        M[this_key] = {'K0':key_top, 'R':R, 'A':m[0], 'B':m[1], 'dt':delta_t, 'sigma':sigma}  
        
        if return_data_est:
            return R, G.dot(m)
        else:
            return R
        
def fit_shifted(delta_t_list, sigma, catalog, WF, M, key_top,  t_tol=None):
    """
    Find the shift value that minimizes the value between a template and a waveform
    """     
    R_dict=dict()
    G=np.ones((WF.p.size, 2))

    if t_tol is None:
        t_tol=WF['t_samp']/10.

    # first search the (coarse) input values of delta_t.  We will refine based on the best of these
    delta_t=delta_t_list.copy()
    delta_t_searched=list()
    # search over delta_t
    while (len(delta_t_searched)==0) or  (np.diff(np.array(delta_t_searched)).min() > t_tol) :       
        for deltaTval in delta_t:
            R_dict[deltaTval]=wf_misfit(deltaTval, sigma, WF, catalog, M,  key_top, G=G)
            bisect.insort(delta_t_searched, deltaTval)
        # make a list of R_vals searched
        R_vals=np.array([R_dict[deltaTval] for deltaTval in delta_t_searched])
        # sort the R_vals
        iR=np.argsort(R_vals)
        # The next search value is the golden-section-rule value (0.62 of the 
        #way between the times for the best and second best residuals)
        delta_t=[((0.62*delta_t_searched[iR[0]]+0.38*delta_t_searched[iR[1]]))]
 
    this_key=key_top+[sigma]+[delta_t_searched[iR[0]]]
    M[key_top+[sigma]]['best']={'key':this_key,'R':R_vals[iR[0]]}
    return R_vals[iR[0]]

def broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top,  t_tol=None):
    """
    Calculate the misfit between a broadened template and a waveform (searching over a range of shifts) 
    """
    this_key=key_top+[sigma]
    if this_key in M:
        return M[this_key]
    else:
        M[this_key]=listDict()
        if this_key not in catalog:
            # if we haven't already broadened the WF to sigma, try it now:
            if sigma==0:
                catalog[this_key]={'t':catalog[key_top].t, 'p':catalog[key_top].p}
            else:    
                nK=3*np.ceil(sigma/WF.dt)
                tK=np.arange(-nK, nK+1)*WF.dt
                K=gaussian(tK, 0, sigma)
                catalog[this_key]=waveform(catalog[key_top].t, np.convolve(catalog[key_top].p, K,'same'))         
        return fit_shifted(delta_ts, sigma, catalog, WF,  M, key_top, t_tol=t_tol) 
 
def fit_broadened(  delta_ts, sigmas, catalog, WF,  M, key_top,  t_tol=None): 
    """
    Find the best broadening value that minimizes the misfit between a template and a waveform
    """     
    R=np.zeros_like(sigmas)
    for ii, sigma in enumerate(sigmas):         
        R[ii]=broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top, t_tol=t_tol)
        if ii>0 and R[ii]>R[ii-1]:
            break
    iR=np.argmin(R[0:ii+1])
    this_key=key_top+[sigmas[iR]]
    M[key_top]['best']={'key':this_key,'R':R[iR]}
    return R[iR]
    
def fit_catalog(WFs, catalog_in, sigmas, delta_ts, t_tol=None, return_data_est=False):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms
    
    Inputs:
        WFs: a waveform object, whose fields include:
            't': the waveform's time vector
            'p': the power samples of the waveform
            'tc': a relative time relative to which the waveform's time is shifted
        catalog_in: A dictionary containing waveform objects that will be broadened and
                    shifted to match the waveforms in 'WFs'                     
        sigmas: a list of spread values that will be searched for each template and waveform
                The search over sigmas terminates when a minimum is found
        delta_ts: a list of time-shift values that will be searched for each template and
                waveform.  All of these will be searched, then the results will be refined 
                to a tolerance of t_tol        
        keyword arguments:
            return_data_est:  set to 'true' if the algorithm should return the best-matching
                shifted and broadened template for each input
            t_tol: tolerance for the time search, defaults to WF.t_samp/10
    Outputs:
        WFp: a set of best-fitting waveform parameters that give:
            delta_t: the time-shift required to align the template and measured waveforms
            sigma: the broadening applied to the measured waveform
            k0: the key into the waveform catalog for the best-fitting waveform
    
    """
    if t_tol is None:
        t_tol=WFs['dt']*0.1
    # make an empty container where we will keep waveforms we've tried already
    catalog=listDict()
    keys=list(catalog_in)
    fit_params=[None for ii in range(WFs.size)]
    # loop over input waveforms
    for WF_count, WF in range(WFs.size):
        WF=WFs[WF_count]
        # set up a matching dictionary (contains keys of waveforms and their misfits)
        M=listDict()
        # loop over the library of templates
        R=np.zeros(len(keys))
        for ii, kk in enumerate(keys):
            # check if we've searched this template before, otherwise copy it into
            # the library of checked templates
            if kk not in catalog:
                catalog[[kk]]=catalog_in[kk].copy()
            if kk not in M:
                M[[kk]]=listDict()
            # find the best misfit between this template and the waveform
            R[ii]=fit_broadened(delta_ts, sigmas, catalog, WF, M, [kk], t_tol=t_tol)
        iR=np.argsort(R)
        this_key=[keys[iR[0]]]
        M['best']={'key':this_key, 'R':R[iR[0]]}
        # recursively traverse the M dict for the best match.  The lowest-level match
        # will not have a 'best' entry
        while 'best' in M[this_key]:
            this_key=M[this_key]['best']['key']
        # write out the best model information 
        fit_params[WF_count]=M[this_key]
        #print(this_key+[R[iR][0]])
        if return_data_est or DOPLOT:
            #             wf_misfit(delta_t, sigma, WF, catalog, M, key_top, G=None, return_data_est=False):
            R0, wf_est=wf_misfit(fit_params[WF_count]['dt'], fit_params[WF_count]['sigma'], WF, catalog, M, [this_key[0]], return_data_est=True)
            fit_params[WF_count]['wf_est']=wf_est
        if DOPLOT:
            plt.figure(); 
            plt.plot(WF.t,WF.p,'k.')
            plt.plot(WF.t, wf_est,'r')
            plt.title('K=%f, dt=%f, sigma=%f, R=%f' % (this_key[0], fit_params[WF_count]['dt'], fit_params[WF_count]['sigma'], fit_params[WF_count]['R']))
            print(WF_count)
    return fit_params
 
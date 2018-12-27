# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import bisect
from waveform import waveform
from time import time
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

def integer_shift(p, delta, fill_value=np.NaN):
    result = np.empty_like(p)
    delta=np.int(delta)
    if delta > 0:
        result[:delta] = fill_value
        result[delta:] = p[:-delta]
    elif delta < 0:
        result[delta:] = fill_value
        result[:delta] = p[-delta:]
    else:
        result[:] = p
    return result

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def lin_fit_misfit(x, y, G=None, m=None, Ginv=None, good_old=None):
    """
    Calculate the best-fitting background + scaled waveform model, return its
    misfit 
    """
    if G is None:
        G=np.ones((x.size, 2))
    G[:,0]=x.ravel()
    good=np.isfinite(G[:,0]) & np.isfinite(y.ravel())
    if good_old is not None and Ginv is not None and np.all(good_old==good):
        # use the previously calculated version of Ginv
        m=Ginv.dot(y[good])
        R=R=np.sqrt(np.sum((y[good]-G[good,:].dot(m))**2.)/(good.sum()-2))
    else:    
        # need at least three good values to calculate a misfit
        if good.sum() < 3:
            m=np.zeros(2)
            R=np.sqrt(np.sum(y**2)/(y.size-2))
            return R, m
        G1=G[good,:]
        try:
            #m=np.linalg.solve(G1.transpose().dot(G1), G1.transpose().dot(y[good]))
            Ginv=np.linalg.solve(G1.transpose().dot(G1), G1.transpose())
            m=Ginv.dot(y[good])
            R=np.sqrt(np.sum((y[good]-G1.dot(m))**2.)/(good.sum()-2))
            #R=np.sqrt(m_all[1][0])
        except np.linalg.LinAlgError:
            m=np.zeros(2)
            R=np.sqrt(np.sum(y**2.)/(y.size-2))
    return R, m, Ginv, good

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
        broadened_key=key_top+[sigma]
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
                broadened_wf=np.convolve(catalog[key_top].p.ravel(), K,'same')
            catalog[broadened_key]=waveform(catalog[key_top].t, broadened_wf, t0=catalog[key_top].t0, tc=catalog[key_top].tc)
        if this_key not in catalog:
            M[this_key]=listDict()
            catalog[this_key] = waveform(catalog[broadened_key].t, \
                   np.interp(WF.t.ravel(), (catalog[key_top].t-catalog[key_top].tc+delta_t).ravel(), broadened_wf.ravel(), left=np.NaN, right=np.NaN), \
                   tc=catalog[broadened_key].tc, t0=catalog[broadened_key].t0)
            catalog[this_key].params['Ginv']=None
            catalog[this_key].params['good']=None
        R, m, Ginv, good = lin_fit_misfit(catalog[this_key].p, WF.p, G=G,\
            Ginv=catalog[this_key].params['Ginv'], good_old=catalog[this_key].params['good'])
        catalog[this_key].params['Ginv']=Ginv
        catalog[this_key].params['good']=good
        M[this_key] = {'K0':key_top[0], 'R':R, 'A':np.float64(m[0]), 'B':np.float64(m[1]), 'delta_t':delta_t, 'sigma':sigma}  
        
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
        
    delta_t_spacing=delta_t_list[1]-delta_t_list[0]
    
    # first search the (coarse) input values of delta_t.  We will refine based on the best of these
    delta_t=delta_t_list.copy()
    delta_t_searched=list()
    shift_count=0  
    max_shift=100
    # search over delta_t
    while (len(delta_t_searched)==0) or  (np.diff(np.array(delta_t_searched)).min() > t_tol) :       
        for deltaTval in delta_t:
            R_dict[deltaTval]=wf_misfit(deltaTval, sigma, WF, catalog, M,  key_top, G=G)
            bisect.insort(delta_t_searched, deltaTval)
        # make a list of R_vals searched
        R_vals=np.array([R_dict[deltaTval] for deltaTval in delta_t_searched])
        # find the minimum of the R vals
        iR=np.argmin(R_vals)
        # choose the next search point.  If the searched picked the minimum or maximum of the
        # time offsets, take a step  of delta_t_spacing to the left or right
        if iR==0:
            delta_t = delta_t_searched[0] - delta_t_spacing
        elif iR==len(delta_t_searched)-1:
            delta_t = delta_t_searched[-1] + delta_t_spacing
        else:
            # if the minimum was in the interior, find the largest gap in the delta_t values
            # around the minimum, and add a point using a golden-rule search
            if delta_t_searched[iR+1]-delta_t_searched[iR] > delta_t_searched[iR]-delta_t_searched[iR-1]:
                # the gap to the right of the minimum is largest: put the new point there
                delta_t = 0.62*delta_t_searched[iR] + 0.38*delta_t_searched[iR+1]
            else:
                # the gap to the left of the minimum is largest: put the new point there
                delta_t = 0.62*delta_t_searched[iR] + 0.38*delta_t_searched[iR-1]
        # need to make delta_t a list so that it is iterable
        delta_t=[delta_t]
        shift_count+=1
        if shift_count > max_shift:
            print("WARNING: too many shifts for %d" % catalog[key_top].shot)
    this_key=key_top+[sigma]+[delta_t_searched[iR]]
    M[key_top+[sigma]]['best']={'key':this_key,'R':R_vals[iR]}
    return R_vals[iR]

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
                catalog[this_key]=waveform(catalog[key_top].t, catalog[key_top].p, t0=catalog[key_top].t0, tc=catalog[key_top].tc)
            else:    
                nK=3*np.ceil(sigma/WF.dt)
                tK=np.arange(-nK, nK+1)*WF.dt
                K=gaussian(tK, 0, sigma)
                catalog[this_key]=waveform(catalog[key_top].t, np.convolve(catalog[key_top].p.ravel(), K,'same'))         
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
    
def fit_catalog(WFs, catalog_in, sigmas, delta_ts, t_tol=None, return_data_est=False, return_catalog=False, catalog=None):
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
    # set a sensible tolerance for delta_t if none is specified
    if t_tol is None:
        t_tol=WFs.dt*0.1
    
    # make an empty output_dictionary
    WFp_empty={f:np.NaN for f in ['K0','R','A','B','delta_t','sigma','t0','Kmin','Kmax','shot']}
    if return_data_est:
        WFp_empty['wf_est']=np.zeros_like(WFs.t)+np.NaN
    
    # make an empty container where we will keep waveforms we've tried already
    if catalog is None:
        catalog=listDict()
    keys=np.sort(list(catalog_in))
    fit_params=[WFp_empty.copy() for ii in range(WFs.size)]
    
    t_center=WFs.t.mean()
    # loop over input waveforms
    for WF_count in range(WFs.size):
        WF=WFs[WF_count]
        if WF.nPeaks > 1:
            continue
        # shift the waveform to put its tc at the center of the time vector
        delta_samp=np.round((WF.tc-t_center)/WF.dt)
        WF.p=integer_shift(WF.p, -delta_samp)
        WF.t0=-delta_samp*WF.dt
        
        # set up a matching dictionary (contains keys of waveforms and their misfits)
        M=listDict()
        # loop over the library of templates
        R=np.zeros(len(keys))
        for ii, kk in enumerate(keys):
            # check if we've searched this template before, otherwise copy it into
            # the library of checked templates
            if [kk] not in catalog:
                # make a copy of the current template
                temp=catalog_in[kk]
                catalog[[kk]]=waveform(temp.t, temp.p, t0=temp.t0, tc=temp.tc)
            if [kk] not in M:
                M[[kk]]=listDict()
            # find the best misfit between this template and the waveform
            R[ii]=fit_broadened(delta_ts, sigmas, catalog, WF, M, [kk], t_tol=t_tol)
            if len(keys) > 1:
                if ii > 3:
                    iR=np.argsort(R[0:ii+1])
                    if np.any(R[iR] > R[iR][0]*(1+2/np.sqrt(WF.t.size))) and iR[0] < ii:
                        break
        R=R[0:ii+1]
        iR=np.argsort(R)
        this_key=[keys[iR[0]]]
        M['best']={'key':this_key, 'R':R[iR[0]]}
        # recursively traverse the M dict for the best match.  The lowest-level match
        # will not have a 'best' entry
        while 'best' in M[this_key]:
            this_key=M[this_key]['best']['key']
        # write out the best model information 
        fit_params[WF_count].update(M[this_key])
        fit_params[WF_count]['delta_t'] -= WF.t0[0]
        fit_params[WF_count]['shot'] = WF.shots[0]
        if len(keys) > 1:
            # find the range of K0 vals whose residuals are not significantly different from the optimum
            R_max=fit_params[WF_count]['R']*(1.+1./np.sqrt(WF.t.size))
            kVals=keys[np.where(R<=R_max)[0]]
            fit_params[WF_count]['Kmin']=kVals.min()
            fit_params[WF_count]['Kmax']=kVals.max()
        #print(this_key+[R[iR][0]])
        if return_data_est or DOPLOT:
            #             wf_misfit(delta_t, sigma, WF, catalog, M, key_top, G=None, return_data_est=False):
            WF.t=WF.t-WF.t0
            R0, wf_est=wf_misfit(fit_params[WF_count]['delta_t'], fit_params[WF_count]['sigma'], WFs[WF_count], catalog, M, [this_key[0]], return_data_est=True)
            fit_params[WF_count]['wf_est']=wf_est#integer_shift(wf_est, -delta_samp)
        if DOPLOT:
            plt.figure(); 
            plt.plot(WF.t, integer_shift(WF.p, delta_samp),'k.')
            plt.plot(WF.t, wf_est,'r')
            plt.title('K=%f, dt=%f, sigma=%f, R=%f' % (this_key[0], fit_params[WF_count]['delta_t'], fit_params[WF_count]['sigma'], fit_params[WF_count]['R']))
            print(WF_count)
        if np.mod(WF_count, 1000)==0 and WF_count > 0:
            print('    N=%d, N_keys=%d' % (WF_count, len(list(catalog))))
    
    result=dict()
    for key in WFp_empty:
        if key is 'wf_est':
            result[key]=np.concatenate( [ ii['wf_est'] for ii in fit_params ], axis=1 )
        else:
            result[key]=np.array([ii[key] for ii in fit_params]).ravel()
    
    if return_catalog:
        return result, catalog
    else:
        return result
 
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

class waveform():
    def __init__(self,p, t0, dt):
        self.p=p
        self.t0=t0
        self.dt=dt
        self.t=None
        self.size=self.p.size
        self.shape=self.p.size
    def get_t(self):
        if self.t is None:
            self.t=self.t0+np.arange(self.p.size)*self.dt
        return self


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

def wf_centroid(WF, els=None):
    """
    Calculate the centroid of a distribution, optionally for the subset specified by "els"
    """
    if els is None:
        els=np.ones_like(WF['p'], dtype=bool)
    return np.sum(WF['t'][els]*WF['p'][els])/WF['p'][els].sum()

def wf_sigma(WF, els=None, C=None):
    """
    Calculate the standard deviation of the energy in a distribution,  optionally for the subset specified by "els"
    """
    if els is None:
        els=np.ones_like(WF['t'], dtype=bool)
    if C is None:
        C=wf_centroid(WF, els)
    return np.sqrt(np.sum(((WF['t'][els]-C)**2)*WF['p'][els])/WF['p'][els].sum())

def wf_percentile(WF, P, els=None):
    """
    Calculate the specified percentiles of a distribution,  optionally for the subset specified by "els"
    """
    if els is not None:
        C=np.cumsum(WF['p'][els])
        return np.interp(P, C/C[-1], WF['t'][els]) 
    else:
        C=np.cumsum(WF['p'])
        return np.interp(P, C/C[-1], WF['t']) 

def wf_robust_spread(WF, els=None):
    """
    Calculate half the difference bewteen the 16th and 84th percentiles of a distribution
    """
    lowHigh=wf_percentile(WF, np.array([0.16, 0.84]), els=els)
    return (lowHigh[1]-lowHigh[0])/2.
    
def nSigmaMean(WF, N=3, els=None, tol=None, maxCount=20):
    """
        Calculate the iterative N-sigma edit, using the robust spread to measure sigma
    """
    if tol is None:
        tol=0.1*(WF['t'][1]-WF['t'][0])
    if els is None:
        els=WF['p']>0
    else:
        els = els & (WF['p'] > 0)
    t_last=WF['t'][0]
    tc=wf_centroid(WF, els)  
    sigma=wf_robust_spread(WF, els)
    count=0
    while (np.abs(t_last-tc) > tol) and (count<maxCount):
        count+=1
        these=(WF['p'] > 0) & (np.abs(WF['t']-tc) < N*sigma)
        t_last=tc;
        tc=wf_centroid(WF, els=these)
        sigma=wf_robust_spread(WF, els=these)
    return tc, sigma
    

def regular_grid_interp_mtx(x0, xi, delta=None):
    """
    Fast linear interpolation matrix script, creates a sparse matrix that, when dotted with a vector of nodal values, returns the interpolated values at the specified data points
    """
    if delta is None:
        delta=x0[1]-x0[0]
    inBds=np.where((xi >= x0[0]) & (xi < x0[-1]))[0]
    ii=(xi[inBds]-x0[0])/delta
    di=ii-np.floor(ii)   
    M=sps.coo_matrix((np.c_[1-di, di].ravel(), \
                      (np.c_[inBds, inBds].ravel(), np.c_[np.floor(ii), np.floor(ii)+1].ravel().astype(int))), shape=(xi.size, x0.size)).tocsr()
    return M, inBds

#def shift_vector(WF0, WF1, delta=0, report_inBds=False):
#    
#    
#    ys=np.zeros_like(WF0['p'])
#    x0=WF0['t0']+WF0['t_samp']*np.arange(WF0['p'].size)
#    x1=delta_t+WF1['t0']+WF1['t_samp']*np.arange(WF1['p'].size)
#    inBds_i=np.where((xi >= x0[0]) & (xi < x0[-1]))[0]
#    ni=inBds_i.size
#    try:
#        delta_ind=(xi[inBds_i[0]]-x0[0])/delta
#    except IndexError:
#        print("here!")
#    W=(delta_ind-np.floor(delta_ind))
#    i0=int(np.floor(delta_ind))
#    ys[inBds_i]=(1-W)*y0[i0:i0+ni]+W*y0[i0+1:i0+ni+1]
#    if report_inBds:
#        return ys, inBds_i
#    else:
#        return ys

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def lin_fit_misfit(x, y, G=None, m=None):
    if G is None:
        G=np.ones((x.size, 2))
    G[:,0]=x.ravel()
    good=np.isfinite(G[:,0]) & np.isfinite(y.ravel())
    G1=G[good,:]
    try:
        #m=np.linalg.inv(G1.transpose().dot(G1)).dot(G1.transpose().dot(y[good]))
        #m_all=np.linalg.lstsq(G1, y[good])
        #m=m_all[0]
        m=np.linalg.solve(G1.transpose().dot(G1), G1.transpose().dot(y[good]))
        R=np.sqrt(np.sum((y[good]-G1.dot(m))**2.))
        #R=np.sqrt(m_all[1][0])
    except np.linalg.LinAlgError:
        m=np.zeros(2)
        R=np.sqrt(np.sum(y**2.))
    #R=np.sqrt(np.sum((y[good]-G1.dot(m))**2.))
    return R, m

def wf_misfit(delta_t, sigma, WF, catalog, M, key_top,  G=None, return_data_est=False):
    if G is None:
        G=np.ones((WF['p'].size, 2))
    this_key=key_top+[sigma]+[delta_t]
    if (this_key in M) and (return_data_est is False):
        return M[this_key]['R']     
    else:
        # check if the broadened but unshifted version of this key is in the catalog
        broadened_key=key_top+[sigma]+[0]
        if broadened_key in catalog:
            broadened_wf=catalog[broadened_key]['p']
        else:
            # make a broadened version of the catalog WF
            if sigma==0:
                 broadened_wf = catalog[key_top]['p']
            else:
                dt=np.diff(catalog[key_top]['t'][0:2])
                nK=3*np.ceil(sigma/dt)
                tK=np.arange(-nK, nK+1)*dt
                K=gaussian(tK, 0, sigma)
                broadened_wf=np.convolve(catalog[key_top]['p'], K,'same')
            catalog[broadened_key]={'p':broadened_wf}
        if this_key not in catalog:
            M[this_key]=listDict()
            catalog[this_key]=catalog[broadened_key].copy()          
            catalog[this_key]['p']=np.interp(WF['t'], catalog[key_top]['t']+delta_t, broadened_wf)
            
        R, m = lin_fit_misfit(catalog[this_key]['p'], WF['p'], G=G)
        M[this_key] = {'K0':key_top, 'R':R, 'A':m[0], 'B':m[1], 'dt':delta_t, 'sigma':sigma}  
        
        if return_data_est:
            return R, G.dot(m)
        else:
            return R
        
def fit_shifted(delta_t_list, sigma, catalog, WF, M, key_top,  t_tol=None):     
    R_dict=dict()
    G=np.ones((WF['p'].size, 2))

    if t_tol is None:
        t_tol=WF['t_samp']/10.
    #if DOPLOT:
    #    plt.clf()
    #    plt.plot(WF['t'], WF['p'],'k')
    # first search the (coarse) input values of delta_t.  We will refine based on the best of these
    delta_t=delta_t_list.copy()
    delta_t_searched=list()
    while (len(delta_t_searched)==0) or  (np.diff(np.array(delta_t_searched)).min() > t_tol) :       
        for t_val in delta_t:
            R_dict[t_val]=wf_misfit(t_val, sigma, WF, catalog, M,  key_top, G=G)
            bisect.insort(delta_t_searched, t_val)
        # make a list of R_vals searched
        R_vals=np.array([R_dict[t_val] for t_val in delta_t_searched])
        # sort the R_vals
        iR=np.argsort(R_vals)
        # The next search value is the golden-rule value (0.7 of the way between 
        #  the times for the best and second best residuals)
        delta_t=[((0.7*delta_t_searched[iR[0]]+0.3*delta_t_searched[iR[1]]))]
 
    this_key=key_top+[sigma]+[delta_t_searched[iR[0]]]
    M[key_top+[sigma]]['best']={'key':this_key,'R':R_vals[iR[0]]}
    return R_vals[iR[0]]

def broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top,  t_tol=None):
    this_key=key_top+[sigma]
    if this_key in M:
        return M[this_key]
    else:
        M[this_key]=listDict()
        if this_key not in catalog:
            # if we haven't already broadened the WF to sigma, try it now:
            if sigma==0:
                catalog[this_key]={'t':catalog[key_top]['t'], 'p':catalog[key_top]['p']}
            else:    
                dt=np.diff(catalog[key_top]['t'][0:2])
                nK=3*np.ceil(sigma/dt)
                tK=np.arange(-nK, nK+1)*dt
                K=gaussian(tK, 0, sigma)
                catalog[this_key]={'t':catalog[key_top]['t'], 'p':np.convolve(catalog[key_top]['p'], K,'same')}         
        return fit_shifted(delta_ts, sigma, catalog, WF,  M, key_top, t_tol=t_tol) 
 
def fit_broadened(  delta_ts, sigmas, catalog, WF,  M, key_top,  t_tol=None):     
    R=np.zeros_like(sigmas)
    for ii, sigma in enumerate(sigmas):         
        R[ii]=broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top, t_tol=t_tol)
        if ii>0 and R[ii]>R[ii-1]:
            break
    iR=np.argmin(R[0:ii+1])
    this_key=key_top+[sigmas[iR]]
    M[key_top]['best']={'key':this_key,'R':R[iR]}
    return R[iR]
    
def fit_catalog(WFs, catalog_in, sigmas, delta_ts, return_data_est=False):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms
    
    Inputs:
        WFs: a list of waveforms.  Each waveform is a dict with entries:
            't0': the waveform's starting time
            't_samp': the waveform's sampling interval
            'p': the power samples of the waveform
        catalog_in: A dictionary containing template that will be broadened and
                    shifted to match the waveforms in 'WFs'.  Each must have entries:
                        't0' : the time of the first sample in the template
                        't_samp' : the sampling interval in the template
                        'p': the power in the template
        sigmas: a list of spread values that will be searched for each template and waveform
        
        delta_ts: a list of time-shift values that will be searched for each template and
                waveform.
        
        keyword argument:
            return_data_est:  set to 'true' if the algorithm should return the best-matching
                shifted and broadened template for each input
    Outputs:
        WFp: a set of best-fitting waveform parameters that give:
            delta_t: the time-shift required to align the template and measured waveforms
            sigma: the broadening applied to the measured waveform
            k0: the key into the waveform catalog for the best-fitting waveform
    
    """
    # make an empty container where we will keep waveforms we've tried already
    catalog=listDict()
    keys=list(catalog_in)
    fit_params=[None for ii in range(len(WFs))]
    # loop over input waveforms
    for WF_count, WF in enumerate(WFs):
        if 't0' in WF:
            t0=WF['t0']
        else:
            t0=0
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
            R[ii]=fit_broadened(delta_ts+t0, sigmas, catalog, WF, M, [kk], t_tol=0.1)
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
            plt.plot(WF['t'],WF['p'],'k.')
            plt.plot(WF['t'], wf_est,'r')
            plt.title('K=%f, dt=%f, sigma=%f, R=%f' % (this_key[0], fit_params[WF_count]['dt'], fit_params[WF_count]['sigma'], fit_params[WF_count]['R']))
            print(WF_count)
    return fit_params
    
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
    
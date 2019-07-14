# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:56:31 2018

@author: ben
"""
import numpy as np
#from numpy.linalg import inv
import matplotlib.pyplot as plt
#import bisect
import sys
from IS2_calval.waveform import waveform
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

def golden_section_search(f, x0, delta_x, bnds=[-np.Inf, np.Inf], integer_steps=False, tol=0.01, max_count=100, refine_parabolic=False):
    """
    iterative search using the golden-section algorithm (more or less)

    Search the points in x0 for the minimum value of f, then either broaden the
    search range by delta_x, or refine the search spacing around the minumum x.
    Repeat until the minimum interval between searched values falls below tol
    If boundaries (bnds) are provided and the search strays outside them, refine
    the spacing around the nearest boundary until the stopping criterion is reached
    """
    # phi is the golden ratio.
    if integer_steps is False:
        phi=(1+np.sqrt(5))/2
    else:
        phi=2
    b=1.-1./phi
    a=1./phi
    # if x0 is not a list or an array, wrap it in a list so we can iterate
    if not hasattr(x0,'__iter__'):
        x0=[x0]
    searched=np.array([])
    R_dict=dict()
    it_count=0
    largest_gap=np.Inf
    while (len(searched)==0) or (largest_gap > tol):
        for x in x0:
            R_dict[x]=f(x)
            searched=np.union1d(searched, [x])
        # make a list of R_vals searched
        R_vals=np.array([R_dict[x] for x in searched])
        # find the minimum of the R vals
        try:
            iR=np.argmin(R_vals)
        except TypeError:
            print("HERE!")
        # choose the next search point.  If the searched picked the minimum or maximum of the
        # time offsets, take a step  of delta_x to the left or right
        if iR==0:
            x0 = [searched[0] - delta_x]
            if x0[0] < bnds[0]:
                # if we're out of bounds, search between the minimum searched value and the left boundary
                x0=[a*bnds[0]+b*np.min(searched[searched > bnds[0]])]
                largest_gap=searched[1]-searched[0]
            else:
                largest_gap=np.Inf
        elif iR==len(searched)-1:
            x0 = [searched[-1] + delta_x ]
            if x0[0] > bnds[1]:
                x0=[a*bnds[1]+b*np.max(searched[searched < bnds[1]])]
                largest_gap=searched[-1]-searched[-2]
            else:
                largest_gap=np.Inf
        else:
            # if the minimum was in the interior, find the largest gap in the x values
            # around the minimum, and add a point using a golden-rule search
            if searched[iR+1]-searched[iR] > searched[iR]-searched[iR-1]:
                # the gap to the right of the minimum is largest: put the new point there
                x0 = [ a*searched[iR] + b*searched[iR+1] ]
                largest_gap=searched[iR+1]-searched[iR]
            else:
                # the gap to the left of the minimum is largest: put the new point there
                x0 = [ a*searched[iR] + b*searched[iR-1] ]
                largest_gap=searched[iR]-searched[iR-1]
        if integer_steps is True:
            if np.floor(x0) not in searched:
                x0=np.floor(x0).astype(int)
            elif np.ceil(x0) not in searched:
                x0=np.ceil(x0).astype(int)
            else:
                break
        # need to make delta_t a list so that it is iterable
        it_count+=1
        if it_count > max_count:
            print("WARNING: too many shifts")
            break
    if refine_parabolic and (len(searched) >= 3):
        return parabolic_search_refinement(searched, R_vals)
    else:
        return searched[iR], R_vals[iR]

def parabolic_search_refinement(x, R):
    """
    Fit a parabola to search results (x, R), return the refined values
    """
    ind=np.argsort(x)
    xs=x[ind]
    Rs=R[ind]
    best=np.argmin(R)
    if best==0:
        p_ind=[0, 1, 2]
    elif best==len(R)-1:
        p_ind=np.array([-2, -1, 0])+best
    else:
        p_ind=best+np.array([-1, 0, 1])
    G=np.c_[np.ones((3,1)), (xs[p_ind]-xs[p_ind[1]]), (xs[p_ind]-xs[p_ind[1]])**2]

    m=np.linalg.solve(G, Rs[p_ind])
    x_opt= -m[1]/2/m[2]
    R_opt= m[0]  + m[1]*x_opt + m[2]*x_opt**2
    return x_opt+xs[p_ind[1]], R_opt

def gaussian(x, ctr, sigma):
    """
        return a normalized gaussian kernel centered on 'ctr' with width 'sigma'
    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-ctr)**2/2/sigma**2)

def amp_misfit(x, y, els=None, A=None):
    """
    misfit for the best-fitting scaled template model.
    """
    if els is None:
        ii=np.isfinite(x) & np.isfinite(y)
    else:
        ii=els
    if A is None:
        A=np.sum(x[ii]*y[ii])/np.sum(x[ii]*x[ii])
    r=y[ii]-x[ii]*A
    R=np.sqrt(np.sum((r**2)/(ii.sum()-2)))
    return R, A, ii

def lin_fit_misfit(x, y, G=None, m=None, Ginv=None, good_old=None):
    """
    Misfit for the the best-fitting background + scaled template model
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

def wf_misfit(delta_t, sigma, WF, catalog, M, key_top,  G=None, return_data_est=False, fit_BG=False):
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
            broadened_p=catalog[broadened_key].p
        else:
            # make a broadened version of the catalog WF
            if sigma==0:
                 broadened_p = catalog[key_top].p
            else:
                nK=3*np.ceil(sigma/WF.dt)
                tK=np.arange(-nK, nK+1)*WF.dt
                K=gaussian(tK, 0, sigma)
                broadened_p=np.convolve(catalog[key_top].p.ravel(), K,'same')
            catalog[broadened_key]=waveform(catalog[key_top].t, broadened_p, t0=catalog[key_top].t0, tc=catalog[key_top].tc)
        # check if the shifted version of the broadened waveform is in the catalog
        if this_key not in catalog:
            M[this_key]=listDict()
            catalog[this_key] = waveform(catalog[broadened_key].t, \
                   np.interp(WF.t.ravel(), (catalog[key_top].t-catalog[key_top].tc+delta_t).ravel(), broadened_p.ravel(), left=np.NaN, right=np.NaN), \
                   tc=catalog[broadened_key].tc, t0=catalog[broadened_key].t0)
            catalog[this_key].params['Ginv']=None
            catalog[this_key].params['good']=None
        if fit_BG is True:
            # solve for the background and the amplitude
            R, m, Ginv, good = lin_fit_misfit(catalog[this_key].p, WF.p, G=G,\
                Ginv=catalog[this_key].params['Ginv'], good_old=catalog[this_key].params['good'])
            catalog[this_key].params['Ginv']=Ginv
            catalog[this_key].params['good']=good
            M[this_key] = {'K0':key_top[0], 'R':R, 'A':np.float64(m[0]), 'B':np.float64(m[1]), 'delta_t':delta_t, 'sigma':sigma}
        else:
            # solve for the amplitude only

            G=catalog[this_key].p
            try:
                ii=np.where(G>0.002)[0][0]
            except IndexError:
                print("The G>002 problem happened!")
            good=np.isfinite(G) & np.isfinite(WF.p)
            good[0:np.maximum(0,ii-4)]=0
            R, m, good = amp_misfit(G, WF.p, els=good)
            M[this_key] = {'K0':key_top[0], 'R':R, 'A':m, 'B':0., 'delta_t':delta_t, 'sigma':sigma}
        if return_data_est:
            return R, G.dot(m)
        else:
            return R

def fit_shifted(delta_t_list, sigma, catalog, WF, M, key_top,  t_tol=None, refine_parabolic=True):
    """
    Find the shift value that minimizes the misfit between a template and a waveform

    performs a golden-section search along the time dimension
    """
    G=np.ones((WF.p.size, 2))
    if t_tol is None:
        t_tol=WF['t_samp']/10.
    delta_t_spacing=delta_t_list[1]-delta_t_list[0]
    bnds=[np.min(delta_t_list)-5, np.max(delta_t_list)+5]
    fDelta = lambda deltaTval: wf_misfit(deltaTval, sigma, WF, catalog, M,  key_top, G=G)
    delta_t_best, R_best = golden_section_search(fDelta, delta_t_list, delta_t_spacing, bnds=bnds, tol=t_tol, max_count=100, refine_parabolic=refine_parabolic)
    this_key=key_top+[sigma]+[delta_t_best]
    if this_key not in M:
        R_best=fDelta(delta_t_best)
    #M[this_key]={'key':this_key, 'R':R_best, 'sigma':sigma, 'delta_t':delta_t_best}
    M[key_top+[sigma]]['best']={'key':this_key, 'R':R_best, 'delta_t':delta_t_best}
    return R_best

def broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top,  t_tol=None, refine_parabolic=True):
    """
    Calculate the misfit between a broadened template and a waveform (searching over a range of shifts)
    """
    this_key=key_top+[sigma]
    if (this_key in M) and ('best' in M[this_key]):
        return M[this_key]['best']['R']
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
                K=K/np.sum(K)
                catalog[this_key]=waveform(catalog[key_top].t, np.convolve(catalog[key_top].p.ravel(), K,'same'))
        return fit_shifted(delta_ts, sigma, catalog, WF,  M, key_top, t_tol=t_tol, refine_parabolic=refine_parabolic)

def fit_broadened(delta_ts, sigmas, WF, catalog,  M, key_top, sigma_tol=None, sigma_max=5., t_tol=None, sigma_last=None):
    """
    Find the best broadening value that minimizes the misfit between a template and a waveform
    """
    if key_top not in M:
        M[key_top]=listDict()
    fSigma = lambda sigma:broadened_misfit(delta_ts, sigma, WF, catalog, M, key_top, t_tol=t_tol)
    sigma_step=2*sigma_tol
    FWHM2sigma=2.355
    if sigmas is None:
        sigma_template=catalog[key_top].fwhm()[0]/FWHM2sigma
        sigma_WF=WF.fwhm()[0]/FWHM2sigma
        sigma0=sigma_step*np.ceil(np.sqrt(np.maximum(0,  sigma_WF**2-sigma_template**2))/sigma_step)
        dSigma=np.maximum(sigma_step, np.ceil(sigma0/4.))
        sigmas=np.array([0., np.maximum(sigma_step, sigma0-dSigma), np.maximum(sigma_step, sigma0+dSigma)])
    else:
        dSigma=np.max(sigmas)[0]/4.
    if np.any(~np.isfinite(sigmas)):
        print("NaN in sigma for %d " % WF.shots)
    if sigma_tol is None:
        sigma_tol=.125
    if sigma_last is not None:
        i1=np.maximum(1, np.argmin(np.abs(sigmas-sigma_last)))
    else:
        i1=1
    sigma_list=[sigmas[0], sigmas[i1]]
    sigma_best, R_best = golden_section_search(fSigma, sigma_list, dSigma, bnds=[0, sigma_max], tol=sigma_tol, max_count=20)
    this_key=key_top+[sigma_best]
    M[key_top]['best']={'key':this_key,'R':R_best}
    return R_best

def fit_catalog(WFs, catalog_in, sigmas, delta_ts, t_tol=None, sigma_tol=None, return_data_est=False, return_catalog=False, catalog=None):
    """
    Search a library of waveforms for the best match between the broadened, shifted library waveform
    and the target waveforms

    Inputs:
        WFs: a waveform object, whose fields include:
            't': the waveform's time vector
            'p': the power samples of the waveform
            'tc': a center time relative to which the waveform's time is shifted
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
    if sigma_tol is None:
        sigma_tol=0.25
    # make an empty output_dictionary
    WFp_empty={f:np.NaN for f in ['K0','R','A','B','delta_t','sigma','t0','Kmin','Kmax','shot']}
    if return_data_est:
        WFp_empty['wf_est']=np.zeros_like(WFs.t)+np.NaN

    # make an empty container where we will keep waveforms we've tried already
    if catalog is None:
        catalog=listDict()
    keys=np.sort(list(catalog_in))

    # loop over the library of templates
    for ii, kk in enumerate(keys):
        # check if we've searched this template before, otherwise copy it into
        # the library of checked templates
        if [kk] not in catalog:
            # make a copy of the current template
            temp=catalog_in[kk]
            catalog[[kk]]=waveform(temp.t, temp.p, t0=temp.t0, tc=temp.tc)

    W_catalog=np.zeros(keys.shape)
    for ind, key in enumerate(keys):
        W_catalog[ind]=catalog_in[key].fwhm()[0]

    fit_params=[WFp_empty.copy() for ii in range(WFs.size)]
    sigma_last=None
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
        # this is the bulk of the work, and it's where problems happen.  Wrap it in a try:
        # and write out errors to be examined later
        if True:
            if len(keys)>1:
                 # Search over input keys to find the best misfit between this template and the waveform
                fB=lambda ind: fit_broadened(delta_ts, None,  WF, catalog, M, [keys[ind]], sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last)
                W_match_ind=np.where(W_catalog >= WF.fwhm()[0])[0]
                if len(W_match_ind) >0:
                    ind=np.array(tuple(set([0, W_match_ind[0]-2,  W_match_ind[0]+2])))
                    ind=ind[(ind >= 0) & (ind<len(keys))]
                else:
                    ind=[2, 4]
                iBest, Rbest = golden_section_search(fB, ind, delta_x=2, bnds=[0, len(keys)-1], integer_steps=True, tol=1)
                iBest=int(iBest)
            else:
                # only one key in input, return its misfit
                Rbest=fit_broadened(delta_ts, None, WF, catalog, M, [keys[0]], sigma_tol=sigma_tol, t_tol=t_tol, sigma_last=sigma_last)
                iBest=0
            this_key=[keys[iBest]]
            M['best']={'key':this_key, 'R':Rbest}
            searched_keys = np.array([this_key for this_key in keys if [this_key] in M])
            R=np.array([M[[ki]]['best']['R'] for ki in searched_keys])

            # recursively traverse the M dict for the best match.  The lowest-level match
            # will not have a 'best' entry
            while 'best' in M[this_key]:
                this_key=M[this_key]['best']['key']
            # write out the best model information
            fit_params[WF_count].update(M[this_key])
            fit_params[WF_count]['delta_t'] -= WF.t0[0]
            fit_params[WF_count]['shot'] = WF.shots[0]
            sigma_last=M[this_key]['sigma']
            R_max=fit_params[WF_count]['R']*(1.+1./np.sqrt(WF.t.size))
            if np.sum(searched_keys>0)>=3:
                these=np.where(searched_keys>0)[0]
                if len(these) > 3:
                     ind_keys=np.argsort(R[these])
                     these=these[ind_keys[0:4]]
                E_roots=np.polynomial.polynomial.Polynomial.fit(np.log10(searched_keys[these]), R[these]-R_max, 2).roots()
                if np.any(np.imag(E_roots)!=0):
                    fit_params[WF_count]['Kmax']=10**np.minimum(3,np.polynomial.polynomial.Polynomial.fit(np.log10(searched_keys[these]), R[these]-R_max, 1).roots()[0])
                    fit_params[WF_count]['Kmin']=np.min(searched_keys[R<R_max])
                else:
                    fit_params[WF_count]['Kmin']=10**np.min(E_roots)
                    fit_params[WF_count]['Kmax']=10**np.max(E_roots)
            if (0. in searched_keys) and R[searched_keys==0]<R_max:
                fit_params[WF_count]['Kmin']=0.

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
        #except KeyboardInterrupt:
        #    sys.exit()
        #except Exception as e:
        #    print("Exception thrown for shot %d" % WF.shots)
        #    print(e)
        #    pass
        if np.mod(WF_count, 1000)==0 and WF_count > 0:
            print('    N=%d, N_keys=%d' % (WF_count, len(list(catalog))))

    result=dict()
    for key in WFp_empty:
        if key in ['wf_est']:
            result[key]=np.concatenate( [ ii['wf_est'] for ii in fit_params ], axis=1 )
        else:
            result[key]=np.array([ii[key] for ii in fit_params]).ravel()

    if return_catalog:
        return result, catalog
    else:
        return result

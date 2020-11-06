# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:40:52 2018
@author: ben
"""

import numpy as np

def unique_by_rows(x, return_dict=False, return_index=False, return_inverse=False):
    """
    determine the unique rows in an array
    
    inputs:
        x:  array of values
        return_dict (default false): return a dictionary whose keys are the 
            unique values of x (unsorted) and whose values are vectors of row indices containing those values
        return_index: Return an index such that x[ind]==uX
        return_inverse: Return an inverse index such that uX[ind]==x
    output:
        uX: array whose rows are the unique rows of x
        ...plus other optional return values...
    """    
    ind=np.zeros(x.shape[0])
    # bin the data by the values in each column.  From left to right, the importance
    # of the value to the sorting decreases by a factor the number of distinct 
    # values in each column
    scale=1.
    if len(x.shape)==1:
        x.shape=[x.shape[0], 1]
    for col in range(x.shape[1]):        
        z, ii=np.unique(x[:,col].astype(np.float64), return_inverse=True)
        scale /= (np.max(ii).astype(float)+1.)
        ind += ii * scale     
    u_ii, index, inverse=np.unique(ind, return_index=True, return_inverse=True)
    uX=x[index,:]
    
    if return_dict is True:
        bin_dict={}
        inv_arg_ind=np.argsort(inverse)
        inv_sort=inverse[inv_arg_ind]
        ind_delta=np.concatenate([[-1], np.where(np.diff(inv_sort))[0], [inv_sort.size]])
        for ii in range(len(ind_delta)-1):
            this_ind=inv_arg_ind[(ind_delta[ii]+1):(ind_delta[ii+1]+1)]
            bin_dict[tuple(x[this_ind[0], :])]=this_ind
        return uX, bin_dict
    if return_index is True and return_inverse is False:
        return uX, index
    elif return_index is False and return_inverse is True:
        return uX, inverse
    elif return_inverse is True and return_index is True:
        return uX, index, inverse
    else:
        return uX

if False:
    # test code
    import matplotlib.pyplot as plt
    W=10
    x=np.random.rand(1000)*W
    y=np.random.rand(x.size)*W
    bin_dict=unique_by_rows(np.c_[np.round(x),np.round(y)], return_dict=True)
    plt.figure()
    for ctr in bin_dict:
        hl=plt.plot(x[bin_dict[ctr]], y[bin_dict[ctr]])
        hs=plt.plot(ctr[0], ctr[1],'*', color=hl[0].get_color())
        ho=plt.plot(x[bin_dict[ctr]][-1], y[bin_dict[ctr]][-1],'o', color=hl[0].get_color())
    # should see squiggles in different colors,  each contained in a 1x1 box    
        
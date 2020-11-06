# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:56:05 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from unique_by_rows import unique_by_rows
import pointCollection as pc
import glob
import re
import h5py

def read_xovers(xover_dir, verbose=False, wildcard='*', r_limits=[0, 1.e7], delta_t_limit=2592000):
    """
    read_xovers: Read all the crossover files in a directory (or matching a glob)

    Inputs:
        xover_dir: directory to search
        verbose: set to 'true' to see errors reading crossover files (default is silent)
        wildcard: default is '*'.  Specify to match selected files
        r_limits: limits for the polar stereographic coordinates of the tile files.  
            Default of [0, 1e7] eliminates crossovers with lat=0 (a common error in early versions)  
        delta_t_limit: set to limit time difference of crossovers, in seconds.  Default of 2592000 is 1 month

    Outputs:
        v: dict of nx2 matrices, giving ATL06 parameters interpolated to the crossover locations.  The first
            column gives the value for the first measurement in the crossover, the second the value from the second.
        delta: dict of nx2 matrices, giving ATL06 parameter differences between the crossover measurents, late minus early
        meta: metadata values at the crossovers
    """


    tiles=glob.glob(xover_dir+'/*.h5')
    print(len(tiles))
    with h5py.File(tiles[0],'r') as h5f:
        fields=[key for key in h5f['data_0'].keys()]
    
    D=[]
    meta=[]

    tile_re=re.compile('E(.*)_N(.*).h5')
    for tile in glob.glob(xover_dir+'/'+wildcard+'.h5'):
        m=tile_re.search(tile)
        if m is not None:
            r2=np.float(m.group(1))**2+np.float(m.group(2))**2
            if (r2 < r_limits[0]**2) or (r2 > r_limits[1]**2):
                continue
        try:
            this_D=[pc.data().from_h5(tile, field_dict={gr : fields}) for gr in ['data_0','data_1']]
            this_meta=pc.data().from_h5(tile, field_dict={None:['slope_x', 'slope_y','grounded']})
            if delta_t_limit is not None:
                good=np.abs(this_D[1].delta_time[:,0]-this_D[0].delta_time[:,0]) < delta_t_limit
                for Di in this_D:
                    Di.index(good)
                this_meta.index(good)

        except KeyError:
            if verbose:
                print("failed to read " + tile)
            continue

        D.append(this_D)
        meta.append(this_meta)

    meta=pc.data().from_list(meta)
    v={}
    for field in fields:
        vi=[]
        for Di in D:
            vi.append(np.r_[[np.sum(getattr(Di[ii], field)*Di[ii].W, axis=1) for ii in [0, 1]]])
        v[field]=np.concatenate(vi, axis=1).T
    delta={field:np.diff(v[field], axis=1) for field in fields}
    bar={field:np.mean(v[field], axis=1) for field in fields}
    return v,  delta,  bar, {key:getattr(meta, key) for key in meta.fields}



# main code.  Parse the only input argument (the crossover directory)

xover_dir=sys.argv[1]
ATL06_fields=['BP','LR','W','cycle_number','rgt','h_li','h_li_sigma','x','y','spot']

v,  delta,  bar, meta = read_xovers(xover_dir)#, wildcard='E400_N-500')
meta['slope_mag']=np.abs(meta['slope_x']+1j*meta['slope_y'])

# error analysis to pick out the worst RGTs:
DD={}
sign=[1, -1]
dh_dict={}

# build a dict dh values, Each entry in the dict is an array of all the delta-h measurements including a particular rgt and cycle combination
# The dh values are multiplied by -1 for the first measurement in a crossover
for col in [0, 1]:
    # loop over combinations of cycles and tracks in each column
    u_ct, D_ct = unique_by_rows(np.c_[np.round(v['rgt'][:, col]).astype(int), np.round(v['cycle_number'][:,col]).astype(int)], return_dict=True)
    for ct in u_ct:
        key=tuple(ct)
        if key not in dh_dict:
            dh_dict[key]=[]
        these=D_ct[key]
        good=(bar['atl06_quality_summary'][these]<0.01) & (meta['slope_mag'][these]<0.02)
        if np.any(good):
            dh_dict[key].append(delta['h_li'][these[good]]*sign[col])

# loop over the entries in dh_dict (combinations of rgt and cycle).  Calculate the median and mad of the differences
D2={}
for key in dh_dict:
    if len(dh_dict[key]) ==0:
        continue
    dh=np.concatenate([item.ravel() for item in dh_dict[key]])
    D2[key]={}
    D2[key]['med']=np.nanmedian(dh)
    D2[key]['count']=dh.size
    D2[key]['mad']=np.nanmedian(np.abs(dh-np.nanmedian(dh)))
    D2[key]['good_count']=np.nansum(np.abs(dh)<1).astype(np.float)
    if np.abs(D2[key]['med']) > 1 or D2[key]['mad'] > 5:
        print("HERE!")

# accumulate the meds, mads, and counts
keys=list(D2.keys())
meds=np.array([D2[key]['med'] for key in keys])
mads=np.array([D2[key]['mad'] for key in keys])
counts=np.array([D2[key]['count'] for key in keys])
good_counts=np.array([D2[key]['good_count'] for key in keys])
rgts=np.array([key[0] for key in keys])
mads[counts<100]=np.NaN
meds[counts<100]=np.NaN
good_counts[counts<100]=np.NaN

# report rgt/cycle combos with med > 1 m or mad > 5 m if there are at least 10 good measurements
med_threshold = 0.5
mad_threshold = 5
count_threshold = 10

print("List of all RGT/cycle combinations with median difference > 0.5 m or MAD difference > 5 m:")  

for ii in range(len(keys)):
    if ((np.abs(meds[ii])>1) or (mads[ii]>5)) and (good_counts[ii]>10):
        print('cycle=%d, rgt=%d, med=%3.1f, mad=%3.1f, N=%d' % (keys[ii][1], keys[ii][0], meds[ii], mads[ii], counts[ii]))
        #print("%d,%d" % (int(keys[ii][0]), int(keys[ii][1])))

fig=plt.figure();
ax=fig.add_subplot(121)
ax.hist(np.log10(mads), 25)
ax.set_xlabel('log10(MAD, m)')
ax=fig.add_subplot(122)
ax.hist(meds, 25)
ax.set_xlabel('med, m')

fig=plt.figure();
ax=fig.add_subplot(111)
ax.plot(rgts, meds, 'k', label='medians')
ax.plot(rgts, meds-mads, 'r--', label='medians +- mads')
ax.plot(rgts, meds+mads, 'r--')
ax.legend()
ax.set_xlabel('rgt')
ax.set_ylabel('xover differece, m')

print("list of rgts with MAD > 0.08 m")
print(rgts[np.abs(mads)>0.08])

plt.show()




 

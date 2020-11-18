#! /usr/bin/env python3

import h5py
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pointCollection as pc
import glob

def read_one_hist(thefile, spot_max):

    IPF={}
    IPF_sigma={}
    Nbar={}
    Ntot={}
    bins=None
    zbin=None
    z=None
    with h5py.File(thefile,'r') as h5f:
        for pair in ['1', '2', '3']:
            for lr in ['l','r']:
                ds=f'gt{pair}{lr}'
                try:
                    if bins is None:
                        bth=np.array(h5f['/gt1l/residual_histogram']['bin_top_h'])
                        bin_w=np.zeros_like(bth)
                        bin_w[1:]=np.diff(bth)
                        bin_w[0] = 1
                        zbin=bth-bin_w/2
                        bins=np.abs(zbin)<5
                        z=zbin[bins]
                    spot=np.float64(h5f[ds].attrs['atlas_spot_number'])
                    N=np.array(h5f[f'/{ds}/residual_histogram']['count']).astype(float)
                    N[N>10000]=0
                    Ntot[spot]=np.nansum(N[:,bins], axis=1)
                    good=Ntot[spot] > spot_max[int(spot)]/4
                    Ntot[spot]=Ntot[spot][good]
                    Nbar[spot] = np.nanmean(N[good,:][:,bins], axis=0)
                    IPF[spot] = Nbar[spot] / bin_w[bins]
                    IPF_sigma[spot] = np.sqrt(np.nansum(N[good,:][:,bins], axis=0))/(np.sum(bins)*bin_w[bins])
                    
                except Exception:
                    pass
                
    return z, Ntot, Nbar, IPF, IPF_sigma
    
out_dir=sys.argv[1]
files=sys.argv[2:]

if files[0] == '--queue':
    files=[]
    D={}
    for thedir in glob.glob('/css/icesat-2/ATLAS/ATL06.003/2*'):
        for thefile in glob.glob(thedir+'/ATL06*11_003_*.h5'):
            files+= [thefile]
    with open('res_hist_queue.txt','w') as fh:
        count=0
        while count < len(files):
            this_str = f'source activate IS2; python3 collect_mean_res_hist.py {out_dir}'
            for ii in range(20):
                try:
                    this_str += " "+files[count]
                except Exception:
                    pass
                count += 1
            this_str += "\n"
            fh.write(this_str)
    sys.exit()


N_max={'weak':57./2.*10.*3., 'strong':57./2.*10.*12.}
#print(N_max)
spot_max={spot:N_max['weak'] for spot in [2, 4, 6]}
spot_max.update({spot:N_max['strong'] for spot in [1, 3, 5]})
for file in files:
    out_file=out_dir+os.path.basename(file).replace('.h5','_rh_avg.h5')
    z, Ntot, Nbar, P, Ps = read_one_hist(file, spot_max)
    print(out_file)
    for spot in range(1, 7):
        try:
            Ds={'z':z, 'Nbar':Nbar[spot], 'P':P[spot],'P_sigma':Ps[spot]}
            temp=pc.data().from_dict(Ds).to_h5(out_file, group=f'spot_{spot}', replace=False)
        except Exception as e:
            print(e)
            pass
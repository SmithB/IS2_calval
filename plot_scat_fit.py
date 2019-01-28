# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:28:59 2019

@author: ben
"""

#from ATL11.pt_blockmedian import pt_blockmedian
#from ATL11.blockmax import blockmax
from smooth_surf_fit.point_data import point_data
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from is2_calval.fit_ATM_scat import proc_RX
      

GL_proj4='+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
field_dict={None:('latitude','longitude','K0', 'shot')}


fig=plt.figure()
hax=list()
hax.append(fig.add_subplot(121, aspect='equal'))
hax.append(fig.add_subplot(122, aspect='equal', sharex=hax[0], sharey=hax[0]))

#dirs=glob.glob('*/fits')
np.seterr(invalid='ignore')

dirs=['fits']
for count, dir in enumerate(dirs):
    D_list=list()   
    files=glob.glob(dir+'/*.h5')
    for file in files:
        try:
            D_list.append(point_data(columns=0, field_dict=field_dict).from_file(file))
        except OSError:
            pass

    for count1, D in enumerate(D_list):
        # check if this D is valid, delete it if not
        #if not hasattr(D,'latitude'):
        #    D_list.delete(count)
        #    continue
        D.N=np.ones_like(D.K0, dtype=int)*int(count1)
        D.list_of_fields.append('N')


    D=point_data(list_of_fields=D_list[0].list_of_fields).from_list(D_list)
    D.get_xy('+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    these=D.K0==0
    hax[count].plot(D.x[these], D.y[these], 'k.', zorder=0)
    hax[count].set_title(file)

    these=D.K0 > 0
    hs=hax[count].scatter(D.x[these], D.y[these], c=np.log10(D.K0[these]), linewidth=0, vmin=-5, vmax=-2, zorder=1)
    plt.colorbar(hs, ax=hax[count])
    
    xy0=(-382902.1976828752, 970571.1546361502);
    
    hax[0].set_xlim(xy0[0]+np.array([-50, 50]))
    hax[0].set_ylim(xy0[1]+np.array([-50, 50]))
    ii=np.where((np.abs(D.x-xy0[0])<20) & (np.abs(D.y-xy0[1])<20))[0]
    shots=[np.min(D.shot[ii]), np.max(D.shot[ii])]
    N=[np.min(D.N[ii]), np.max(D.N[ii])]
    print(shots)
    fit_file=files[N[0]]
    WF_file=os.path.basename(fit_file)
    scat_file='/Users/ben/git_repos/is2_calval/subsurface_srf_no_BC.h5'
    D_out=proc_RX(wf_file, shots, sigmas=np.arange(0, 5, 0.25), deltas=np.arange(-1, 1.5, 0.5), \
        TX_file=fit_file, scat_file=scat_file)[0]
#    while True:
#        xy=plt.ginput()
#        print(xy)
#    

plt.show()

    
    
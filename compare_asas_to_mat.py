# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:44:01 2018

@author: ben
"""
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from ATL11.ATL06_data import ATL06_data

base='/Volumes/ice2/ben/scf/AA_06/'
ASAS_dir=base+'/ASAS/'
mat_dir=base+'/mat_h5_npz'

id_re=re.compile('ATL06_(\d+_\d+)')

mat_files=glob.glob(mat_dir+'/*.h5')
mat_dict={id_re.search(f).group(1):f for f in mat_files}

ASAS_dict={id_re.search(f).group(1):f for f in glob.glob(ASAS_dir+'/*.h5')}

matches=ASAS_dict.keys() & mat_dict.keys()

for match in matches:
    plt.figure();
    
    D6a=ATL06_data(filename=ASAS_dict[match], beam_pair=2)
    D6a.h_li[D6a.h_li>2.e4]=np.NaN
    
    D6m=ATL06_data(filename=mat_dict[match], beam_pair=2)
    
    plt.clf()
    ax1=plt.subplot(3, 1, 1)
    plt.plot(D6a.segment_id, D6a.h_li)
    
    ax2=plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    plt.plot(D6m.segment_id, D6m.h_li)
    
    ax3=plt.subplot(3,1,3, sharex=ax1)
    D6a.index(np.where(np.in1d(D6a.segment_id[:,1], D6m.segment_id[:,1]))[0])
    D6m.index(np.where(np.in1d(D6m.segment_id[:,1], D6a.segment_id[:,1]))[0])
    plt.plot(D6a.segment_id, D6a.h_li-D6m.h_li,'.')
    ax3.set_ylim([-10, 10])
    print()
 
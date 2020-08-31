# -*- coding: utf-8 -*-
"""
Compare ATM data with ICESat-2 data.  

This script uses the geo_index framework to load spatially overlapping blocks of 
ATM and ATL06 data, and calculate differences between ATL06 segments and the
ATM data that measured the same surface patch.

Created on Wed Sep  5 13:36:08 2018

@author: ben
"""

import pointCollection as pc
#from PointDatabase import ATL06_filters
from ATL11.RDE import RDE
import matplotlib.pyplot as plt
from osgeo import osr
import numpy as np
import h5py
import os
import sys
#from ATL11.pt_blockmedian import pt_blockmedian
from PointDatabase import pt_blockmedian

import matplotlib.pyplot as plt
DOPLOT=False
VERBOSE=False

def my_lsfit(G, d):
    try:
        # this version of the inversion is much faster than linalg.lstsq
        m=np.linalg.solve(G.T.dot(G), G.T.dot(d))#, rcond=None)
        #m=m0[0]
    except ValueError:
        print("ValueError in LSq")
     
    r=d-G.dot(m)
    R=np.sqrt(np.sum(r**2)/(d.size-G.shape[1]))
    sigma_hat=RDE(r)
    return m, R, sigma_hat

def blockmedian_for_qsub(Qdata, delta):
    # make a subset of the qfit data that contains the blockmedian elevation values    
    
    lat0=np.nanmedian(Qdata.latitude)
    lon0=np.nanmedian(Qdata.longitude)
    # calculate the ellipsoid radius for the current point
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)
    delta_lon=np.mod(Qdata.longitude-lon0+180.,360.)-180 
    # project the Qfit latitude and longitude into northing and easting
    EN=Re*np.c_[delta_lon*np.cos(d2r*lat0), (Qdata.latitude-lat0)]*np.pi/180.
    Qz=Qdata.elevation.astype(np.float64)
    xm, ym, zm, ind=pt_blockmedian(EN[:,0], EN[:,1], Qz, delta, return_index=True);#, randomize_evens=True)
    
    for field in Qdata.fields:
        temp=getattr(Qdata, field)
        setattr(Qdata, field, 0.5*(temp[ind[:,0]]+temp[ind[:,1]]))
    Qdata.longitude=0.5*(delta_lon[ind[:,0]]+delta_lon[ind[:,1]])+lon0
    return Qdata

def compare_seg_with_qfit(D6i, Qdata, beam, out_template):
    
    # calculate the ellipsoid radius for the current point
    lat0=D6i.latitude[beam]
    lon0=D6i.longitude[beam]
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)
    
    # project the Qfit latitude and longitude into northing and easting
    EN=Re*np.c_[(np.mod(Qdata.longitude-lon0+180.,360.)-180)*np.cos(d2r*lat0), (Qdata.latitude-lat0)]*np.pi/180.
    
    # take a 50-meter circular subset
    ind_50m=np.sum(EN**2,axis=1)<50**2
    if np.sum(ind_50m) < 10:
        return None
    EN=EN[ind_50m,:]
    z=Qdata.elevation[ind_50m].astype(np.float64)
    
    # calculate along-track vector and the across-track vector
    this_az=D6i.seg_azimuth[beam]
    if not np.isfinite(this_az):
        return None
    at_vec=np.array([np.sin(this_az*d2r), np.cos(this_az*d2r)])
    xt_vec=at_vec[[1,0]]*np.array([-1, 1])
    
    # project the Qfit data into the along-track coordinate system
    xy_at=np.c_[np.dot(EN, at_vec), np.dot(EN, xt_vec)]

    # copy the ATL06 values into the output template                                     
    this_out=out_template.copy()                
    copy_fields=['segment_id','x','y', 'dh_fit_dx', 'h_li','h_li_sigma', 'h_mean','orbit_number',
                 'atl06_quality_summary', 'w_surface_window_final','n_fit_photons', 'fpb_n_corr',
                 'delta_time', 'h_robust_sprd', 'snr_significance','y_atc','rgt','dac','spot']

    for field in copy_fields:
        this_out[field]=getattr(D6i, field)[beam]
    
    #fit a plane to all data within 50m of the point
    G=np.c_[np.ones((ind_50m.sum(), 1)), xy_at]
    if not np.all(np.isfinite(G.ravel())):
        print("Bad G")
    try:
        m, R, sigma_hat=my_lsfit(G, z)
    except TypeError:
        return None
    this_out['sigma_qfit_50m']=R
    this_out['h_qfit_50m']=m[0]
    this_out['dh_qfit_dx']=m[1]
    this_out['dh_qfit_dy']=m[2]
    this_out['N_50m']=np.sum(ind_50m)
    this_out['dz_50m']=D6i.h_li[beam]-m[0]
    this_out['RDE_50m']=sigma_hat
    
    ind_20m=np.sum(EN**2,axis=1)<20**2
    if np.sum(ind_20m)> 5:
        this_out['hbar_20m']=np.nanmean(z[ind_20m])
    
    sub_seg=np.logical_and(np.abs(xy_at[:,1])<5, np.abs(xy_at[:,0])<30)
    if np.sum(sub_seg)<10:
        return None
    G=np.c_[np.ones((sub_seg.sum(), 1)), xy_at[sub_seg,0]]
    m, R, sigma_hat=my_lsfit(G, z[sub_seg])
    this_out['sigma_qfit_seg']=R
    this_out['h_qfit_seg']=m[0]
    this_out['N_seg']=np.sum(sub_seg)
    this_out['dz_seg']=D6i.h_li[beam]-m[0]
    this_out['RDE_seg']=sigma_hat
    this_out['beam']=beam
    this_out['beam_pair']=D6i.BP[beam]
    this_out['t_qfit']=np.nanmean(Qdata.days_J2k[ind_50m])
    this_out['y_seg_mean']=np.nanmean(xy_at[sub_seg,1])
    this_out['x_seg_mean']=np.nanmean(xy_at[sub_seg,0])
 



sigma_pulse=5.5

ATL06_dir=sys.argv[1]
ATM_dir=sys.argv[2]

print("working on ATL06 dir %s, ATM directory %s" % (ATL06_dir,  ATM_dir))

Qfit_fields=['latitude','longitude','elevation','rel_time']
 
Qfit_index=ATM_dir+'/GeoIndex_1km.h5'
out_dir=ATM_dir+'/xovers'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
out_file='%s/vs_%s.h5' %(out_dir, ATL06_dir.replace('/index','').replace('/', '_'))

if os.path.isfile(out_file):
    print("found: "+out_file)

ATL06_index='/Volumes/ice2/ben/scf/AA_06/'+ATL06_dir+'/GeoIndex.h5'

SRS_proj4='+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
ATL06_field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','segment_id','sigma_geo_h','atl06_quality_summary'], 
            'ground_track':['x_atc', 'y_atc','seg_azimuth','sigma_geo_at','sigma_geo_xt'],
            'geophysical':['dac'],
            'bias_correction':['fpb_n_corr'],
            'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','dh_fit_dy','h_mean', 'h_rms_misfit','h_robust_sprd','n_fit_photons','w_surface_window_final','snr_significance'],
            'orbit_info':['rgt','orbit_number'],
            'derived': ['BP','spot']}

ATL06_fields=list()
for key in ATL06_field_dict:   
    ATL06_fields+=ATL06_field_dict[key]
            
WGS84a=6378137.0
WGS84b=6356752.31424
d2r=np.pi/180.
delta=[10000., 10000.]
  
Q_GI=pc.geoIndex(SRS_proj4=SRS_proj4).from_file(Qfit_index, read_file=True)

D6_GI=pc.geoIndex(SRS_proj4=SRS_proj4).from_file(ATL06_index, read_file=True)
# the qfit index is at 1 km , the ATL06 index is at 10 km. find the overlap between the two
# Interesect the ATL06 index with the Qfit index

xy_10km_Q=Q_GI.bins_as_array()
xy_10km_Q=np.unique(xy_10km_Q[0]+1j*xy_10km_Q[1])
D6_GI=D6_GI.copy_subset(xyBin=[np.real(xy_10km_Q), np.imag(xy_10km_Q)]);

out_fields=[ 
    'segment_id','x','y','beam', 'BP', 'h_li', 'h_li_sigma', 'atl06_quality_summary', 
    'dac', 'rgt','orbit_number','spot',
    'dh_fit_dx','N_50m','N_seg','h_qfit_seg','dh_qfit_dx','dh_qfit_dy', 
    'h_robust_sprd', 'snr_significance',
    'h_qfit_50m','sigma_qfit_50m', 'sigma_seg','dz_50m','E_seg','RDE_seg',
    'hbar_20m',
    'RDE_50m','t_seg','t_qfit','y_atc', 'x_seg_mean', 'y_seg_mean']
out_template={f:np.NaN for f in out_fields}
out=list()
 
#READ IN THE ATM DATA HERE AND BUILD THE QTREE
# ALSO RUN BLOCKMEDIAN_FOR_QSUB

#plt.figure(1)
for bin_name in D6_GI.keys():
    #plt.clf()
    print(bin_name)
    bin_xy=[int(coord) for coord in bin_name.split('_')]
    # query the Qfit index to get all the data for the current bin.  pad=6 means that we read 6 km on either side of the central bin, covering the 10-km ATL06 bin plus 1 km buffer
    Qlist=Q_GI.query_xy([[bin_xy[0]], [bin_xy[1]]], pad=6, get_data=True, strict=True,  fields=Qfit_fields)
    Qsub=pc.data().from_list(Qlist).get_xy(SRS_proj4)
    Qsub=blockmedian_for_qsub(Qsub, 5)
    
    
    #START CUTTING HERE
    
    # the geo index works much better (more selective) if the Qfit data are sorted by geobin
    
    x0=np.round(Qsub.x/100.)*100
    y0=np.round(Qsub.x/100.)*100
    iB=np.argsort(x0+(y0-y0.min())/(y0.max()-y0.min()))
    Qsub.index(iB)
    # index the sorted Qfit data
    GI_Qsub=pc.geoIndex(delta=[100, 100]).from_xy([Qsub.x, Qsub.y])
    
    # query ATL06 for the current bin, and index it    
    D6list=D6_GI.query_xy([[bin_xy[0]], [bin_xy[1]]], get_data=True, fields=ATL06_field_dict)
    if not isinstance(D6list, list):
        D6list=[D6list]
        
    KK=ATL06_field_dict.copy()
    D6sub=pc.ATL06.data().from_list(D6list).get_xy(SRS_proj4)
    x0=np.round(np.nanmean(D6sub.x, axis=1)/100.)*100
    y0=np.round(np.nanmean(D6sub.y, axis=1)/100.)*100
    iB=np.argsort(x0+(y0-y0.min())/(y0.max()-y0.min()))
    D6sub.index(iB)
    # index the sorted ATL06 data at 100 m
    GI_D6sub=pc.geoIndex(delta=[100, 100]).from_xy([np.nanmean(D6sub.x, axis=1), np.nanmean(D6sub.y, axis=1)])
    
    # find the common bins at 100 m
    GI_D6sub, GI_Qsub=GI_D6sub.intersect(GI_Qsub, pad=[0, 1])
    if GI_D6sub is None:
        print("no intersections found for ATL06 bin %s\n" % (bin_name))
        continue
    
    # loop over 100-meter bins in the ATL06 data
    for bin100 in GI_D6sub:
        #print("\t"+bin100)
        # grab the Qfit bins around the ATL06 bin
        this_GI_Qsub=GI_Qsub.copy_subset(xyBin=[int(item) for item in bin100.split('_')], pad=1)
        # and subset the ATL06 data to the current bin
        D6sub2=D6sub.copy_subset(np.arange(GI_D6sub[bin100]['offset_start'],GI_D6sub[bin100]['offset_end']+1, dtype=int), by_row=True)
        #print("\t\tbefore:%d" % len(out))
        for i_AT in range(D6sub2.latitude.shape[0]):
            D6i=D6sub2.copy_subset(i_AT)
            qQ=this_GI_Qsub.query_xy([np.nanmean(D6i.x), np.nanmean(D6i.y)], pad=2, get_data=False)
            if qQ is None:
                continue
            # subset the qfit data for the current segment
            Qlist=list()
            for ii in qQ.keys():
                for iStart, iEnd in zip(qQ[ii]['offset_start'], qQ[ii]['offset_end']):
                    Qlist.append(Qsub.copy_subset(np.arange(iStart, iEnd, dtype=int)))
            Qdata=pc.data().from_list(Qlist)
            Qdata.index(np.isfinite(Qdata.elevation) & np.isfinite(Qdata.latitude) & np.isfinite(Qdata.longitude))
            for beam in [0,1]:
                if not (np.isfinite(D6i.h_li[beam]) &  np.isfinite(D6i.latitude[beam]) & np.isfinite(D6i.longitude[beam])):
                    continue
#STOP CUTTING                
                this_out=compare_seg_with_qfit(D6i, Qdata, beam, out_template)
                if this_out is not None:
                    out.append(this_out)
            

D=dict()
with h5py.File(out_file,'w') as h5f:
    for field in out[0].keys():
        D[field]=np.array([ii[field] for ii in out])
        h5f.create_dataset(field, data=D[field])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 09:02:21 2018

@author: ben
"""
import numpy as np
from IS2_calval.read_DEM import read_DEM
from ATL11.pt_blockmedian import pt_blockmedian
import matplotlib.pyplot as plt
import osgeo
import scipy.interpolate as sI
import scipy.stats as sps
import argparse
import h5py
import sys
import os
from ATL11.ATL06_data import ATL06_data
import ATL11.ATL06_filters as f06

def validate_xi(xy, xy0):
    # identify points that are inside an interpolation grid
    good=np.ones_like(xy[0], dtype=bool)
    for dim in [0,1]:
        good=good & (xy[dim] >= xy0[dim][0]) & (xy[dim] <= xy0[dim][-1])
    return good

def queryIndex(index_file, demFile, verbose=False):
    from ATL11.geo_index import geo_index
    from IS2_calval.demBounds import demBounds
  
    pointData=dict()
    beam_names=['l','r']
    field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude'], 
                    'fit_statistics':['dh_fit_dx', 'h_rms_misfit','h_robust_sprd','n_fit_photons', 'signal_selection_source','snr_significance','w_surface_window_final'],
                    'derived':['valid']}
    gI=geo_index().from_file(index_file)
    xr, yr = demBounds( demFile, proj4=gI.attrs['SRS_proj4'] )
    xr += np.array([-1e4, 1e4])
    yr += np.array([-1e4, 1e4])
    xy=gI.bins_as_array()
    plt.figure(); plt.plot(xy[0], xy[1],'k.')
    plt.plot(xr[[0, 1, 1, 0, 0]], yr[[0, 0, 1, 1, 0]],'r')
    D6es=gI.query_xy_box( xr, yr, get_data=True, fields=field_dict)
    if verbose:
        temp=gI.query_xy_box( xr+np.array([-1e4, 1e4]), yr+np.array([-1e4, 1e4]), get_data=False)
        print("list of datasets:")
        for key in set(temp.keys()):             
            print("\t%s"% key)
    for D6 in D6es:
        f06.phDensityFilter(D6, toNaN=True, subset=True)
        if D6.h_li.size==0:
            continue
        f06.segDifferenceFilter(D6, tol=2, toNaN=True, subset=True)
        if D6.h_li.size==0:
            continue
        for beam in [0, 1]:
            if np.sum(np.isfinite(D6.h_li[:, beam])) > 100:
                first=np.max(np.where(np.isfinite(D6.latitude[:,beam]))[0])
                last=np.min(np.where(np.isfinite(D6.latitude[:,beam]))[0])
                AD=np.sign(D6.latitude[last, beam]-D6.latitude[first, beam])
                this_name="%s:gt%d%s" % (os.path.basename(D6.file), D6.beam_pair, beam_names[beam])
                
                pointData[this_name]={'latitude':D6.latitude[:, beam], 'longitude':D6.longitude[:,beam],'h':D6.h_li[:,beam],'delta_time':D6.delta_time[:, beam],'AD':AD+np.zeros_like(D6.delta_time), 'orbit':D6.orbit+np.zeros_like(D6.delta_time)}
             
    return pointData
    
def readPointData(args):
    pointDataSets=dict()
    if args.dem:
        pointData=dict()
        pointData['x'], pointData['y'], pointData['h']=read_DEM(args.pointFile, asPoints=True)
        pointDataSets[args.pointFile]=pointData
    elif args.index is True:
        pointDataSets=queryIndex(args.pointFile, args.demFile, verbose=args.verbose)
    elif args.ATL06 is False and args.ATL08 is False:
        pointData=dict()
        h5f=h5py.File(args.pointFile,'r')
        try:
            pointData['latitude']=np.array(h5f['latitude'])
            pointData['longitude']=np.array(h5f['longitude'])
        except KeyError:
            pointData['x']=np.array(h5f['x'])
            pointData['y']=np.array(h5f['y'])
        pointData['h']=np.array(h5f['h'])
        pointDataSets[args.pointFile]=pointData
        h5f.close()
    elif args.ATL06 is True:        
        field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude'], 
                'fit_statistics':['dh_fit_dx', 'h_rms_misfit','h_robust_sprd','n_fit_photons', 'signal_selection_source','snr_significance','w_surface_window_final', 'h_mean'],
                'derived':['valid']}

        beamPairs=[1, 2, 3]
        beams=['l','r']
        pointData=dict()
        for beamPair in beamPairs:
            pairName='gt%d' % beamPair
            try:
                D6=ATL06_data(beam_pair=beamPair, field_dict=field_dict).from_file(args.pointFile)
                f06.phDensityFilter(D6, toNaN=True, subset=True, minDensity={'weak':0.5, 'strong':2})
                #if D6.h_li.size==0:
                #    continue
                #f06.segDifferenceFilter(D6, tol=10, toNaN=True, subset=True)
                if D6.h_li.size==0:
                    continue
                for ind, beam in enumerate(beams):
                    if D6.longitude.size < 50:
                        continue
                    group=pairName+beam                
                    pointData['longitude']=D6.longitude[:, ind].ravel()
                    pointData['latitude']=D6.latitude[:, ind].ravel()
                    pointData['h']=D6.h_mean[:, ind].ravel()
                    pointData['delta_time']=D6.delta_time[:, ind].ravel()
                    these=np.where(np.isfinite(D6.latitude[:,ind]))[0]
                    if these.size > 2:
                        first=np.max(these)
                        last=np.min(these)
                        pointData['AD']=np.sign(D6.latitude[last, ind]-D6.latitude[first, ind])*np.ones_like(pointData['h'])
                    if pointData['h'].size > 0:
                        pointDataSets[args.pointFile+":"+group]=pointData.copy()
            except KeyError:
                print("pair %s not in %s\n" % (pairName, args.pointFile) )
    if args.blockMedian is not None:
        for key in list(pointDataSets):
            pointDataSets[key]['x'], pointDataSets[key]['y'], pointDataSets[key]['h'] = \
                pt_blockmedian(pointDataSets[key]['x'], pointDataSets[key]['y'], pointDataSets[key]['h'], delta=args.blockMedian)
    # assign the 'name' property of the dataset 
    for key in pointDataSets:
        pointDataSets[key]['name']=key
        
    return pointDataSets
    
def evaluate_shift(dxy, basis_vectors, Dsub, gI, inATC=False, return_shifted=False, iterateTSE=1, minSigma=2, index=None):
    """ 
        evaluate the misfit between altimetry measurements and a DEM 
        for one shift vector
    """
    # returns: R, N, biasSlope, this_delta, dh 
    # define the offset as a function of the basis vectors
    this_delta = dxy[0]*basis_vectors[0] + dxy[1]*basis_vectors[1]
    
    hi=np.zeros_like(Dsub['x'])+np.NaN
    if index is None:
        good=validate_xi((Dsub['y']+this_delta[1], Dsub['x']+this_delta[0]), gI.grid)
    else:
        good=index.copy()
        good[good]=validate_xi((Dsub['y'][good]+this_delta[1], Dsub['x'][good]+this_delta[0]), gI.grid)
    # interpolate the DEM values at the offset data points
    hi[good]=gI((Dsub['y'][good]+this_delta[1], Dsub['x'][good]+this_delta[0]))
    # calculate the difference between the data values and the interpolated values
    dh=Dsub['h']-hi
    ii=np.isfinite(dh)

    if inATC:
        # Solve for the residual slope in the along-track direction
        xATC=(Dsub['y']-Dsub['y'][ii].mean())*basis_vectors[0][1] + \
                (Dsub['x']-Dsub['x'][ii].mean())*basis_vectors[0][0]
        G=np.c_[np.ones(ii.sum()), xATC[ii]/1000]
    else:
        # solve for the best-fitting plane for the residuals
        G=np.c_[np.ones(ii.sum()), \
                (Dsub['x'][ii]-Dsub['x'][ii].mean()).ravel()/1000.,\
                (Dsub['y'][ii]-Dsub['y'][ii].mean()).ravel()/1000.]
    mask=np.ones(G.shape[0], dtype=bool)
    
    dhValid=dh[ii]
    for iteration in range(iterateTSE):
        # solve the normal equations for the model      
        try:          
            biasSlope=np.linalg.solve(G[mask,:].transpose().dot(G[mask,:]), G[mask,:].transpose().dot(dhValid[mask]))  
        except np.linalg.LinAlgError:
            biasSlope=np.zeros(G.shape[1])
            biasSlope[0]=dhValid.mean()
        r=dhValid-G.dot(biasSlope)
        if iterateTSE>1:
            sigma=(sps.scoreatpercentile(r[mask], 84)-sps.scoreatpercentile(r[mask], 16))/2         
            mask=np.abs(r)<3*sigma
        # calculate the mean-squared residual
    R=np.sum(r[mask]**2)/(mask.sum()-G.shape[1]-2)
    if iterateTSE>1:
        ii_ind=np.where(ii)[0]
        ii[ii_ind[mask==0]]=0
    # save the data count
    N=np.sum(ii)
    if return_shifted:
        Dshift=Dsub.copy()
        Dshift['DEM']=np.zeros_like(Dsub['x'])+np.NaN
        Dshift['DEM'][good]=hi[good]
        Dshift['DEM_corr']=np.zeros_like(Dsub['x'])+np.NaN
        biasSlopeNoZero=biasSlope.copy()
        biasSlopeNoZero[0]=0
        Dshift['DEM_corr'][ii]=hi[ii]+G[mask,:].dot(biasSlopeNoZero)
        if inATC:
            Dshift['xATC']=xATC + dxy[0]
        Dshift['x']=Dsub['x']+this_delta[0]
        Dshift['y']=Dsub['y']+this_delta[1]
        Dshift['dh']=np.zeros_like(Dsub['x'])+np.NaN
        Dshift['dh'][ii]=dh[ii]-G[mask,:].dot(biasSlope)
        return R, N, biasSlope, ii, this_delta, Dshift
    else:
        return R, N, biasSlope, ii


def register_DEM(DEM,  projSys, pointData, max_shift=500, delta_initial=50, delta_target=2., inATC=False, DOPLOT=False, lTerrain=None, name=None):
    demRes=DEM['x'][1]-DEM['x'][0]

    # calculate the projection from latlon to the DEM CS
    llRef = osgeo.osr.SpatialReference()
    llRef.ImportFromEPSG(4326)
    demRef = osgeo.osr.SpatialReference()
    demRef.ImportFromWkt(projSys)
    xform = osgeo.osr.CoordinateTransformation(llRef,demRef)
    
    # project the point data into the DEM coordinates
    if 'x' not in pointData:
        xy=np.array(xform.TransformPoints( np.c_[pointData['longitude'], pointData['latitude'], np.zeros_like(pointData['longitude'])]))[:,0:2]
        pointData['x']=xy[:,0]
        pointData['y']=xy[:,1] 
    xyRaw=xy
    # calculate the basis vectors for the offsets
    basis_vectors=list()
    if inATC:
        good_x=np.where(np.isfinite(pointData['x']))[0]
        # if the in_ATC keyword is set, the first basis vector is parallel to the difference between the first and last points
        basis_vectors.append( np.array([pointData['x'][good_x[-1]]-pointData['x'][good_x[0]], \
          pointData['y'][good_x[-1]]-pointData['y'][good_x[0]]]) )
        basis_vectors[0] = basis_vectors[0] / np.sqrt(np.sum(basis_vectors[0]**2))
        # the second vector is perpendicular to the first
        basis_vectors.append(np.array([-basis_vectors[0][1], basis_vectors[0][0]]))
    else:
        # if inATC is false, just use x and y
        basis_vectors.append( np.array([1., 0.]) )
        basis_vectors.append( np.array([0., 1.]) )
    # set up the geo interpolation object
    gI=sI.RegularGridInterpolator((DEM['y'], DEM['x']), DEM['z'], fill_value=np.NaN)
       
    # select the subset of the data that are within the DEM box
    valid=validate_xi((pointData['y'], pointData['x']), gI.grid)
    Dsub=dict()
    for field in pointData.keys():
        if field is not 'name':
            Dsub[field]=pointData[field][valid]
    # next select the subset of the subset that has valid DEM values    
    valid=np.isfinite(gI((Dsub['y'], Dsub['x'])))
    for field in Dsub.keys():
        Dsub[field]=Dsub[field][valid]
    
    if Dsub['x'].size < 10 or (np.max(Dsub['x'])-np.min(Dsub['x'])) < 100*demRes or (np.max(Dsub['y'])-np.min(Dsub['y'])) < 100*demRes :
        print("not enough data to span the offsets")
        return dict(), None, None, xyRaw
   
    # evaluate the lag-zero shift and eliminate the largest residuals
    R0, N0, biasSlope0, validIndex = evaluate_shift(np.array([0., 0.]), basis_vectors, Dsub, gI, inATC, iterateTSE=5)

    if np.max(Dsub['h'][validIndex])-np.min(Dsub['h'][validIndex]) < 10:
        print("not enough topography to allow a match")
        return dict(), None, None, xyRaw
    
    # arrays of values used in shifting data
    dxsub, dysub = np.meshgrid(np.array([-1., 0., 1.]), np.array([-1., 0., 1.]))

    #Run the fit twice.  The first time, using offsets edited using the lag-zero offset, the second time
    # using the best lag from the first round
    # save the h values
    h0=Dsub['h'].copy()  
    
    for iteration in [0, 1]:
        
        # delta is the spacing between test offset values.  It will be reduced iteratively
        delta=delta_initial
        # define a grid of search values that span the initial range (-max_shift to +max_shift)
        [x_offsets, y_offsets]=np.meshgrid(np.arange(-max_shift, max_shift+delta, delta), \
            np.arange(-max_shift, max_shift+delta, delta))

        # copy the stashed value of h back into Dsub
        Dsub['h']=h0.copy()
        Dsub['h'][validIndex==0]=np.NaN
        
        # store the offset values we have searched at the residuals and counts at each in a set of dictionaries
        R=dict()
        N=dict()
        biasSlope=dict()
                    
        # loop until delta has been reduced to delta_target (the /2 means that the last
        # iteration has delta <= delta_target)
        while delta > delta_target/2:
            # Search all offset values defined by the previous iteration
            for dxy in zip(x_offsets.ravel(), y_offsets.ravel()):
                # skip this value if it is already in the R dictionary
                if dxy in R: 
                    continue          
                R[dxy], N[dxy], biasSlope[dxy], validIndexShift = evaluate_shift(dxy, basis_vectors, Dsub, gI, inATC)
            # identify the offsets we've searched
            searched=list(R)
            # calculate the residual values for the keys
            rVals=[R[ii] for ii in searched]
            # find the minimum residual
            off_best=searched[np.argmin(rVals)]
            # define the offsets for the next iteration
            x_offsets=list()
            y_offsets=list()
            # check if the eight points surrounding the current point (+- delta in each direction)
            # have been evaluated.  If not, search them.  If all eight have been searched, 
            # refine delta by a factor of two
            while len(x_offsets) == 0:
                for dxi, dyi in zip( dxsub.ravel(), dysub.ravel()):
                    this_offset=(off_best[0]+dxi*delta, off_best[1]+dyi*delta)
                    if this_offset not in R:
                        # if this offset has not been searched, add it to the list
                        x_offsets.append(this_offset[0])
                        y_offsets.append(this_offset[1])                    
                # if all offsets have been searched, loop to a smaller delta
                if len(x_offsets)==0:
                    delta /= 2
            x_offsets=np.array(x_offsets)
            y_offsets=np.array(y_offsets)
        # collect outputs
        N_vals=np.array([N[ii] for ii in searched])
        xyOff=np.c_[searched]
        # find the best offset
        iBest=np.argmin(rVals)
        delta_xy=np.array(off_best)
        Dsub['h']=h0
        # run this shift again to find the best set of editied residuals
        #R, N, biasSlope, ii, this_delta, Dshift
        R0, N0, BiasSlope0, validIndex, delta0, Dshift0 = evaluate_shift( delta_xy, basis_vectors, Dsub, gI, inATC, return_shifted=True, iterateTSE=5)
        
    if lTerrain is not None:
        # if L_terrain is set, then the number of DOFs used in calculating the errors
        # is the number of unique points at a horizontal scale of L_terrain
        nUnique=len(set([tuple(ii) for ii in np.round(np.c_[Dsub['x']/lTerrain, Dsub['y']/lTerrain])]))
    else:
        nUnique=N_vals[iBest]
    # identify the offsets that are not significantly different from the minimum
    # assuming the errors are independent and normally distributed 
    inEB=np.where(rVals <= rVals[iBest]*(1+np.sqrt(2/nUnique)))[0]
    # sigma_xy is half the range of offsets that are not significantly different
    # from the minimum
    sigma_xy=np.array([(np.max(xyOff[inEB, 0])-np.min(xyOff[inEB, 0]))/2.,\
                       (np.max(xyOff[inEB, 1])-np.min(xyOff[inEB, 1]))/2.])  
  
    R0, N0, BiasSlope0, Valid0, delta0, Dshift0 = evaluate_shift( (0. ,0.), basis_vectors, Dsub, gI, inATC, return_shifted=True)        
    R1, N1, BiasSlope1, Valid1, delta1, Dshift1 = evaluate_shift( delta_xy, basis_vectors, Dsub, gI, inATC, return_shifted=True)        
   
    if DOPLOT:
        plt.figure()
        plt.subplot(311)
        plt.scatter(xyOff[:,0], xyOff[:,1], c=np.sqrt(np.array(rVals)), linewidth=0, \
           vmin=np.sqrt(rVals[iBest]),   vmax=np.sqrt(rVals[iBest]*(1+np.sqrt(2/nUnique))) )
        plt.plot(delta_xy[0], delta_xy[1],'ko',linewidth=2)
        plt.axis('equal')
        plt.colorbar()
        if name is not None:
            plt.title(name)
        
        if inATC:
            plt.subplot(323)
            plt.plot(Dshift0['xATC'], Dshift0['DEM_corr'],'k.')
            plt.plot(Dshift0['xATC'], Dshift0['h'],'r.')
            plt.subplot(324)
            plt.plot(Dshift0['xATC'], Dshift0['dh'],'k.')
        else: 
            vrange=np.array([-2*np.sqrt(R0), 2*np.sqrt(R0)])
            plt.subplot(312)
            plt.scatter(Dsub['x'], Dsub['y'], c=Dshift0['dh'], linewidth=0, vmin=vrange[0], vmax=vrange[1])
            plt.axis('equal')
            plt.colorbar()
            
        if inATC:
            plt.subplot(325)
            plt.plot(Dshift1['xATC'][validIndex], Dshift1['DEM_corr'][validIndex],'k.')
            plt.plot(Dshift1['xATC'][validIndex], Dshift1['h'][validIndex],'r.')
            plt.subplot(326)
            plt.plot(Dshift1['xATC'][validIndex], Dshift1['dh'][validIndex],'k.')
        else:
            plt.subplot(313)
            plt.scatter(Dshift1['x'], Dshift1['y'], c=Dshift1['dh'], linewidth=0, vmin=vrange[0], vmax=vrange[1]) 
            plt.colorbar()        
        
    result={'delta_xy':delta_xy,'sigma_xy':sigma_xy,'R':rVals[iBest],\
            'N':N_vals[iBest],'biasModel': biasSlope[off_best],'basis_vectors':basis_vectors}
    if 'latitude' in Dsub:
        result['lat_mean']=Dsub['latitude'].mean()
        result['lon_mean']=Dsub['longitude'].mean()
    if 'delta_time' in Dsub and Dsub['delta_time'].size > 0:
        result['DOY']=np.nanmean(Dsub['delta_time'])/24/3600
    return result, Dshift0, Dshift1, xyRaw

def main():
    parser=argparse.ArgumentParser(description='Find the best offset for a DEM relative to a set of altimetry data')
    parser.add_argument("demFile", type=str)
    parser.add_argument("pointFile", type=str)
    parser.add_argument('--index', action='store_true')
    parser.add_argument("--ATL06", action="store_true")
    parser.add_argument("--ATL08", action="store_true")
    parser.add_argument("--report_file","-r", type=str, default=None)
    parser.add_argument("--DOPLOT", action="store_true")
    parser.add_argument("--max_offset",'-m', type=float, default=200)
    parser.add_argument("--delta_initial", '-d', type=float, default=40)
    parser.add_argument("--delta_target",'-t', type=float, default=2)
    parser.add_argument("--l_terrain",'-l', type=float, default=None)
    parser.add_argument("--dem", action="store_true")
    parser.add_argument("--inATC", "-i", action="store_true")
    parser.add_argument("--blockMedian", "-b", type=float, default=None)
    parser.add_argument("--verbose",'-v', action="store_true")
    args=parser.parse_args()

    # read the DEM data
    DEM=dict()
    DEM['x'], DEM['y'], DEM['z'], projSys=read_DEM(args.demFile, getProjection=True)
    
    # read the pointdata file
    pointDataSets=readPointData(args)
 
    if args.report_file is not None:
        fid_out=open(args.report_file,'w')
    else:
        fid_out=sys.stdout

    if args.DOPLOT:        
        allD=list()
        allXY=list()
    for dsName in list(pointDataSets):
        result, Dshift0, Dshift1, xyRaw=register_DEM(DEM, projSys, \
                pointDataSets[dsName], max_shift=args.max_offset, delta_initial=args.delta_initial, \
                delta_target=args.delta_target, inATC=args.inATC, DOPLOT=args.DOPLOT,\
                lTerrain=args.l_terrain, name=dsName)     
        if args.DOPLOT:
            allD.append([Dshift0, Dshift1])
            allXY.append(xyRaw)
        if 'delta_xy' not in result:
            continue
        fid_out.write('dataSet = '+dsName+"\n")
        for key in ['orbit','AD']:
            try:
                fid_out.write("%s = %d\n" % (key, np.nanmean(pointDataSets[dsName][key])))
            except KeyError:
                pass
        for key in result:
            if key is "basis_vectors":
                fid_out.write("basis_vector_0 = "+str(result[key][0])+"\n")
                fid_out.write("basis_vector_1 = "+str(result[key][1])+"\n")
            else:
                fid_out.write(key +" = "+str(result[key])+"\n")     
        fid_out.write("\n")
    
    if args.DOPLOT:      
        gx, gy=np.gradient(DEM['z'][::5, ::5])
        Gsigma=sps.scoreatpercentile(np.abs(gx.ravel()[np.isfinite(gx.ravel())]), 95)
        plt.figure()
        plt.imshow(gx, vmin=-Gsigma, vmax=Gsigma, aspect='equal', cmap='Greys', extent=[DEM['x'][0], DEM['x'][-1], DEM['y'][0], DEM['y'][-1]], origin='top')
        for xy in allXY:
            plt.plot(xy[:,0], xy[:,1],'k.',markersize=2)
        for DS in allD:
            if DS is None or DS[1] is None:
                continue
            Dsub=DS[1]
            plt.scatter(Dsub['x'], Dsub['y'], c=Dsub['dh'], linewidth=0, vmin=-1, vmax=1, cmap='bwr')
        plt.show(block=True)
    if args.report_file is not None:
        fid_out.close()    
    return
    
if __name__=="__main__":
    main()
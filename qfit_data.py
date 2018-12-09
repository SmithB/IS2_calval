# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:46:21 2017

Class to read and manipulate ATL06 data.  Currently set up for Ben-style fake data, should be modified to work with the official ATL06 prodct foramt

@author: ben
"""
import h5py
import numpy as np
from datetime import datetime, timedelta
from osgeo import osr
import matplotlib.pyplot as plt 
import re

class Qfit_data:
    np.seterr(invalid='ignore')
    def __init__(self, filename=None, x_bounds=None, y_bounds=None, index_range=[0,-1], field_dict=None, waveform_format=False, list_of_fields=None, list_of_data=None, from_dict=None): 
        if field_dict is None:  
            if waveform_format is False:
                self.waveform_format=False
                field_dict={None:['latitude','longitude','elevation'], 'instrument_parameters':['azimuth','rel_time']} 
            else:
                self.waveform_format=True
                field_dict={'footprint':['latitude','longitude','elevation'],'time':['seconds_of_day']}
        if '__calc_internal__' not in field_dict:
            field_dict['__calc_internal__']=['days_J2K']     
        if list_of_fields is None:
            list_of_fields=list()
            for group in field_dict.keys():
                for field in field_dict[group]:
                    list_of_fields.append(field)

        self.list_of_fields=list_of_fields
        if list_of_data is not None:
            self.from_list(list_of_data)
            return None     
        if from_dict is not None:
            self.list_of_fields=list_of_fields
            for field in list_of_fields:
                setattr(self, field, from_dict[field])
            return
        # read from a file if specified
        if filename is not None:
            # read a list of files if list provided
            if isinstance(filename, (list, tuple)):
                D6_list=[Qfit_data(filename=thisfile, field_dict=field_dict, index_range=index_range, x_bounds=x_bounds, y_bounds=y_bounds) for thisfile in filename]
                self.from_list(D6_list)
            elif isinstance(filename, (str)):
                # this happens when the input filename is a string, not a list
                self.read_from_file(filename, field_dict, index_range=index_range, x_bounds=x_bounds, y_bounds=y_bounds)
            else:
                raise TypeError
        else:
            # no file specified, set blank
            for field in list_of_fields:
                setattr(self, field, np.zeros((2,0)))       
          
    def read_from_file(self, filename, field_dict, index_range=[0,-1], x_bounds=None, y_bounds=None, beam_pair=None, NICK=None): 
        h5_f=h5py.File(filename)
        # find the date and time number in filename
        m=re.search(r"ATM1B.+_(\d\d\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d).*.h5",filename)
        this_time=[int(m.group(ind+1)) for ind in range(6)]
        #for ii, val in enumerate(this_time):
        #    if ii > 4 and this_time[ii] > 59:
        #        this_time[ii] -=60
        #        this_time[ii-1] += 1
        if self.waveform_format:
            t0=datetime(*this_time[0:3])
        else:
            t0=datetime(*this_time[0:3]) + timedelta(hours=this_time[3], minutes=this_time[4], seconds=this_time[5])-datetime(2000, 1, 1, 0, 0, 0)
            t0=t0.days+t0.seconds/24./3600.

        for group in field_dict.keys():
            if group is '__calc_internal__':
                continue
            for field in field_dict[group]:
                if field not in self.list_of_fields:
                    self.list_of_fields.append(field)
                try:
                    if group is None:
                        setattr(self, field, np.array(h5_f[field][index_range[0]:index_range[1]]).transpose())
                    else:
                        setattr(self, field, np.array(h5_f[group][field][index_range[0]:index_range[1]]).transpose())  
                except KeyError:
                    print("could not read %s/%s" % (group, field))
                if field in ['rel_time']:
                    field='days_J2K'
                    if field not in self.list_of_fields:
                        self.list_of_fields.append(field)
                    setattr(self, field, self.rel_time/24/3600.+t0)
        h5_f.close()
        if self.waveform_format:
            self.days_J2K=(t0-datetime(2000,1, 1, 0, 0, 0)).days + self.seconds_of_day/24./3600.
        return
    
    def get_xy(self, proj4_str=None, EPSG=None):
        out_srs=osr.SpatialReference()
        if proj4_str is None and EPSG is not None:
            out_srs.ImportFromProj4(EPSG)
        else:
            out_srs.ImportFromProj4(proj4_str)
        ll_srs=osr.SpatialReference()
        ll_srs.ImportFromEPSG(4326)
        ct=osr.CoordinateTransformation(ll_srs, out_srs)
        #x, y, z = list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(D.longitude), np.ravel(D.latitude), np.zeros_like(np.ravel(D.latitude)))]))
        if self.latitude.size==0:
            self.x=np.zeros_like(self.latitude)
            self.y=np.zeros_like(self.latitude)
        else:
            x, y, z= list(zip(*[ct.TransformPoint(*xyz) for xyz in zip(np.ravel(self.longitude), np.ravel(self.latitude), np.zeros_like(np.ravel(self.latitude)))]))
            self.x=np.reshape(x, self.latitude.shape)
            self.y=np.reshape(y, self.longitude.shape)
        if 'x' not in self.list_of_fields:
            self.list_of_fields.append('x')
            self.list_of_fields.append('y')
        return self
    
    def append(self, D):
        for field in self.list_of_fields:
            setattr(self, np.c_[getattr(self, field), getattr(D, field)])
        return        

    def from_list(self, D_list):
        try:
            for field in self.list_of_fields:
                data_list=[getattr(this_D, field) for this_D in D_list]       
                setattr(self, field, np.concatenate(data_list, 0))
        except TypeError:
            for field in self.list_of_fields:
                setattr(self, field, getattr(D_list, field))
        return self
    
    def index(self, index):
        for field in self.list_of_fields:
            setattr(self, field, getattr(self, field)[index])
        return self
        
    def subset(self, index, by_row=True, datasets=None):
        dd=dict()
        if datasets is None:
            datasets=self.list_of_fields
        for field in datasets:
            temp_field=self.__dict__[field]
            if temp_field.ndim ==1:
                dd[field]=temp_field[index]
            else:
                if by_row is not None and by_row:
                    dd[field]=temp_field[index,:]
                else:
                    dd[field]=temp_field.ravel()[index]
        return Qfit_data(from_dict=dd, list_of_fields=datasets)
            
    def copy(self):
        return Qfit_data(list_of_data=(self), list_of_fields=self.list_of_fields)
  
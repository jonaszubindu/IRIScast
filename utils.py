#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:07:11 2021

@author: jonaszbinden
"""
import sys
if '/sml/zbindenj/vae' not in sys.path:
    sys.path.append('/sml/zbindenj/vae')
if '/sml/zbindenj/features' not in sys.path:
    sys.path.append('/home/zbindenj/features')
import os

from astropy import constants as c


from irisreader import observation, raster_cube
from irisreader.utils.date import to_epoch
from irisreader.utils.date import from_Tformat

from irispy.utils import get_interpolated_effective_area

import irisreader as ir

import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
import sunpy.io
from sunpy import log
from sunpy.io.file_tools import UnrecognizedFileTypeError
from sunpy.time import is_time_in_given_format, parse_time
from sunpy.timeseries.timeseriesbase import GenericTimeSeries
from sunpy.util.metadata import MetaDict
from sunpy.visualization import peek_show
from tqdm import tqdm
import torch
import pickle
import numpy as np
import pandas as pd

# from dask import delayed
# import dask.array as da
# import dask.dataframe as dd

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.interpolate import interp1d
from sklearn import preprocessing
import astroscrappy
from sunpy.coordinates.sun import earth_distance

# import datetime
from datetime import datetime, timedelta, time
from warnings import warn
from utils_features import *
import h5py
import gc
from astropy import constants as c
import logging

import vae

import importlib

DN2PHOT = {'NUV':18/(u.pixel), 'FUV': 4/(u.pixel)}

#######################################################################################################################


""" Utility Functions from Brandon """
def extract_flare_starttime( obs, warn_margin_arcsec=150, margin_minutes=1 ):
    closest = get_closest_mxflare( obs )
    if closest is None:
        raise Exception("No C/M/X flare found! No start time could be recovered. Please check your data.")
    return from_Tformat( closest['event_starttime'] ) - timedelta( minutes=margin_minutes )

from warnings import warn
def get_closest_mxflare( obs, warn_margin_arcsec=150 ):
    mx_flares = obs.goes.events.get_flares(classes="CMX")
    if len( mx_flares ) > 0:
        closest = mx_flares.iloc[0]
        if closest['hpc_x'] == 0 or closest['hpc_y'] == 0:
            warn( "Warning: The closest C/M/X flare is not located correctly in HEK - please check the data manually." )
        elif closest['dist_arcsec'] > warn_margin_arcsec:
            warn( "Warning: The closest C/M/X flare seems to be too far away - please check HEK data for this observation." )
        return closest
    else:
        return None


def find_preflare_end( obs, warn_margin_arcsec=150, margin_minutes=1 ):

    # make sure this works also if keep_null = False

    # get event_starttime of closest flare
    flare_starttime = extract_flare_starttime( obs, warn_margin_arcsec, margin_minutes )

    # get timestamps from raster only at the first raster position
    ts = obs.raster("Mg II k").get_timestamps( raster_pos=0 ) # careful using this single raster position

    # compute difference to flare starttime in seconds
    diffs = np.array( ts ) - to_epoch( flare_starttime )

    # compute first start of the raster sweep before the flare (only before the flare is allowed)
    raster_sweeps_before = np.argmin( np.abs( diffs[diffs<0] ) ) #careful with computations like this!

    return raster_sweeps_before


def time_cut( obs, flare=True, length_minutes=30 ):

    raster = obs.raster("Mg II k")
    ts = raster.get_timestamps( raster_pos=0 )

    if flare:
        stop = find_preflare_end( obs )
        diffs = np.array( ts[:stop] ) - (ts[stop] - length_minutes*60)
        start = np.argmin( np.abs( diffs ) )

    else:
        start = 0
        diffs = (np.array(ts)-ts[0]) - length_minutes*60
        stop = np.argmin( np.abs( diffs ) )

    headers = raster.get_raster_pos_headers( raster_pos=0 )
    delta_t = from_Tformat( headers[stop]['DATE_OBS'] ) - from_Tformat( headers[start]['DATE_OBS'] )
    return [start, stop, delta_t.seconds/60]


def clean_pre_interpol(X, threshold, mode='zeros'):
    '''
    X must have shape (some value, n_breaks)
    clean profiles which meet the following criterion:
        - Negative values -> condition = profile with any value <= -100
        - Noisy -> condition = profile with maximum <= 10 before normalizing
        - Overexposure -> condition = more than 3 points == max


    input  - X: matrix (example, feature)
           - modes:
               1) 'ones' -> replaces bad profiles with a vector of ones
               2) 'nans' -> replaces bad profiles with a vector of nan's
               3) 'del' -> deletes bad profiles (this changes dimentional structure, but usefull for unsupervised)
    '''

    Y1 = np.copy(X) # keeps rejected spectra for checking
    Y2 = np.copy(X) # keeps rejected spectra for checking

    # Clean profiles with negative
    neg = np.min(X, axis=-1)
    #Clean profiles with too little signal
    small = np.max(X, axis=-1)
    # Clean overexposed profiles
    maxx = np.max(X, axis=-1).reshape(small.shape + (1,))
    diff = (X-maxx)
    w = np.where(diff == 0)
    dump = np.zeros(X.shape)
    dump[w] = 1
    s =  np.sum(dump, axis = -1)
#     len(X[X.clip(min=10)>10])
    if mode == 'zeros':

#         spectra_quick_look(X, 1399.5, 1403.8, 344)
        X[np.where(np.all(X  < threshold, axis=-1))] = np.full((1, X.shape[-1]), fill_value = 0)
#         spectra_quick_look(X, 1399.5, 1403.8, 344)
#         keep = []
#         for elem in set(np.where(X.clip(min=10) > 10)[0]):
#             if list(np.where(X.clip(min=10) > 10)[0]).count(elem)>3:
#                 keep.append(elem)
#         inds = [ind for ind in range(X.shape[0]) if ind not in keep]
#         X[inds] = np.full((1, X.shape[-1]), fill_value = 0) # puts everything to 0 which has not minimum 3 points above
                                                            # threshold
        X[np.logical_or(neg <= -100, s >=3)] = np.full((1, X.shape[-1]), fill_value = 0)

#         X[np.where(neg <= -100)] = np.full((1, X.shape[-1]), fill_value = 0) # change in how to clean spectra for special
                                                                             # processing

#         Y1[np.logical_and(neg > -100, s < 10)] = np.full((1, X.shape[-1]), fill_value = 0)
#         Y1 = Y1[~np.all(Y1 == 0, axis=-1)]
#         Y2[np.where(np.any(X >= threshold, axis=-1))] = np.full((1, X.shape[-1]), fill_value = 0)
#         Y2[keep] = np.full((1, X.shape[-1]), fill_value = 0)
#         Y2 = Y2[~np.all(Y2 == 0, axis=-1)]

    if mode == 'del':
        X = np.delete(X, np.concatenate((np.where(neg <= -100)[0], np.where(s>=3)[0])), axis=0) # needs major updating if this is ever used

#     Y = np.vstack([Y1, Y2])
    return X#, Y


def clean_aft_interpol(X, mode='del'):

    """
    This cleaning process is used to reject spectra with cosmic rays and abnormal peak ratio. This needs to be updated when other
    lines will be processed.

    input  - X: matrix (example, feature)
           - modes:
               1) 'ones' -> replaces bad profiles with a vector of ones
               2) 'nans' -> replaces bad profiles with a vector of nan's

    """
    try:
        X = X/X.unit
    except Exception:
        pass

    Y3 = np.copy(X) # keeps rejected spectra for checking
    Y4 = np.copy(X) # keeps rejected spectra for checking
    Y5 = np.copy(X) # keeps rejected spectra for checking

    # Reject spectra outside of peak ratio [1:1, 2:1]
    df_features = extract_features_MgIIk(X, save_path=None)
    inds = df_features.index[(df_features['kh_ratio'] < .8) | (df_features['kh_ratio'] > 2) ] #.7 to efficiently remove noisy peaks, remove nan trip emission (only few) | (np.isnan(df_features['trip_emiss']))

    # Find maximum and compare to the peak window to check if there are spikes outside of the peak window
    maxx_ind = [np.argmax(prof) for prof in X]
    inds_m = np.array([n for n, max_ind in enumerate(maxx_ind) if (max_ind not in np.arange(kl,kr+1,1) and max_ind not in
                                                                   np.arange(hl,hr+1,1))])
    #delete spectra with extreamly high pseudocontinium probably from limb obs
    high_continuum = np.squeeze(np.argwhere(np.mean(X[:,460:500], axis=-1) > .6))


    if mode == 'zeros':

        if high_continuum.size != 0:
            X[high_continuum] = np.full((1, X.shape[-1]), fill_value = 0)
            if np.any(X != 0):
                not_high_continuum = np.array([i for i in range(X.shape[0]) if i not in high_continuum])
                if not_high_continuum.size != 0:
                    Y3[not_high_continuum] = np.full((1, X.shape[-1]), fill_value = 0)
                else:
                    Y3 = Y3[~np.all(Y3 == 0, axis=1)]
        else:
            Y3 = np.zeros_like(X)

        if inds_m.size != 0:
            X[inds_m] = np.full((1, X.shape[-1]), fill_value = 0)
            if np.any(X != 0):
                not_inds_m = np.array([i for i in range(X.shape[0]) if i not in inds_m])
                if not_inds_m.size != 0:
                    Y4[not_inds_m] = np.full((1, X.shape[-1]), fill_value = 0)
                else:
                    Y4 = Y4[~np.all(Y4 == 0, axis=1)]
        else:
            Y4 = np.zeros_like(X)


        if inds.size != 0:
            X[inds] = np.full((1, X.shape[-1]), fill_value = 0)
            if np.any(X != 0):
                not_inds = np.array([i for i in range(X.shape[0]) if i not in inds])
                if not_inds.size != 0:
                    Y5[not_inds] = np.full((1, X.shape[-1]), fill_value = 0)
                else:
                    Y5 = Y5[~np.all(Y5 == 0, axis=1)]
        else:
            Y5 = np.zeros_like(X)

    if mode == 'del':

        del_inds = np.hstack([low_continuum, inds_m, inds])

        X = np.delete(X, del_inds.astype(int), axis=0)

#     print(X, Y3, Y4, Y5)
    if (np.all(Y3 == X) and np.all(Y4 == X)) or mode == 'del':
        Y = []
    else:
        Y = np.vstack([Y3, Y4, Y5])

#     print("func called!")
    return X, Y


def normalize( X ):
    '''
    normalize each profile by its maximum
    '''
    maxx_u = np.max( X, axis=-1 )
    maxx_u = maxx_u.reshape(maxx_u.shape + (1,))

    try:
        X = X/X.unit
    except Exception:
        pass
    maxx = np.max( X, axis=-1 ).reshape(maxx_u.shape)

    X[np.where(np.all(X == 0, axis=-1))] = np.full((1, X.shape[-1]), fill_value = 0)
    X[np.where(np.any(X != 0, axis=-1))] = X[np.where(np.any(X != 0, axis=-1))]/maxx[np.where(np.any(X != 0, axis=-1))]


    return X, maxx_u

def minmaxnorm(X, min_, max_): # apply after standardization on all data
    """mapping the intensity levels onto a [0,1] in a log shape"""
    return (X-min_)/(max_-min_)


def spectra_quick_look(spectra, lambda_min=None, lambda_max=None, n_breaks=None, dim=5, ind012 = None):
    '''
    plot a random sample spectral data
    '''
    if lambda_min:
        lambda_units = lambda_min + np.arange(0,n_breaks)*(lambda_max-lambda_min)/n_breaks
    else:
        lambda_units = np.arange(0,spectra.shape[-1])

    if spectra.shape[0]<=0 or spectra.shape[1]<=0:
        warn(f'spectra can not be rendered because shapes are {spectra.shape[0]} and {spectra.shape[1]}')
        return None
    else:
        if len(spectra.shape)>4:
            spectra = np.vstack(np.vstack(np.vstack(spectra)))
        elif len(spectra.shape)>3:
            spectra = np.vstack(np.vstack(spectra))
        elif len(spectra.shape)>2:
            spectra = np.vstack(spectra)

        if ind012:
            spectra = spectra[ind012, :]
        else:
#             inds_greate_0 = np.where(np.any(spectra>0, axis=-1))
#             spectra = spectra[inds_greate_0]
            ind_select = np.random.randint(0, spectra.shape[0], size=dim*dim) # low >= high outputted
            spectra = spectra[ind_select,:]

        fig = plt.figure(figsize=(36,16))
        gs = fig.add_gridspec(dim, dim, wspace=0, hspace=0)
        for i in range(dim):
            for j in range(dim):
                ind = (i*dim)+j
                ax = fig.add_subplot(gs[i, j])
                ax.grid(False)
                try:
                    plt.plot(lambda_units, spectra[ind], linewidth=1, linestyle='-', color='snow')
                except Exception:
                    plt.plot(spectra[ind], linewidth=1, linestyle='-', color='snow')
                ax.set_facecolor('xkcd:black')
                plt.xticks(fontsize=16, color='red')
                plt.yticks(fontsize=16, color='red')

                #plt.legend()
        plt.show()

    return ind012

##########################################################################################

def wait_for_enter(msg='Press ENTER to continue.'):
    """
    misc.wait_for_enter(msg='Press ENTER to continue.')

    Print a message and wait until the user presses the ENTER key.

    INPUT:
    msg (optional): message

    OUTPUT:
    (none)
    """

    print ('\a') # get user attention using the terminal bell

    input(msg)

def ask_for_value(msg='Enter value = '):
    """
    x = misc.ask_for_value(msg='Enter value = ')

    Print a message asking the user to enter something, wait until the user presses the ENTER key, and return the value.

    INPUT:
    msg (optional): message

    OUTPUT:
    x: user value (string)
    """

    print ('\a') # get user attention using the terminal bell

    x = input(msg)

    return x

# def time_clipping_no_flare(raster, minutes = 60, raster_pos = 0, num_of_partitions_max = 10):

#     """
#     Computes partitions of length minutes for given slit position. The length of the partition will always be >= minutes, if the
#     partition is sparse, this can lead to much longer windows than minutes. Partitions are only created if at least one partition
#     fully fits within the given time window. Partitions between different slits can be highly shifted to each other.

#     If no partitions were found the function returns empty lists.
#     """

#     ts = raster.get_timestamps(raster_pos = raster_pos )
#     time_in_sec_obs = ts[-1] - ts[0]

#     num_of_partitions = np.int64(np.floor(time_in_sec_obs/(minutes*60)))
#     if num_of_partitions == 0:
#         num_of_partitions = 1
#     if num_of_partitions_max != None:
#         if num_of_partitions > num_of_partitions_max:
#             num_of_partitions = num_of_partitions_max + 1

#     data_start_points = [ts[-1]-((i+1)*minutes*60) for i in range(num_of_partitions-1)]

#     data_end_points = [ts[-1]-(i*minutes*60) for i in range(num_of_partitions-1)]

#     diffs = [np.array(ts) - data_start_points[i] for i in range(num_of_partitions-1)]
#     #obs_end = datetime.fromisoformat(raster.headers[0]['DATE_END'])
#     #diffs = np.array( ts ) - to_epoch(obs_end - timedelta(minutes=minutes))
#     raster_sweep_start = [np.argmin(np.abs( diffs[i])) for i in range(num_of_partitions-1)]

#     # this will maybe not exactly be the time minutes but the time to
#     # then it will just be the whole observation which is okay as well.
#     # The only problem that might arise here is for an LSTM to somehow
#     # find a common timegrid.


#     diffs = [np.array(ts) - data_end_points[i] for i in range(num_of_partitions-1)]
#     raster_sweep_end = [np.argmin(np.abs( diffs[i])) for i in range(num_of_partitions-1)]

#     raster_sweep_start = np.sort(raster_sweep_start)
#     raster_sweep_end = np.sort(raster_sweep_end)

#     return raster_sweep_start, raster_sweep_end


def Balance(X1, X2):
    '''
    Takes in two numpy arrays and under samples the larger array to have a 1:1 ratio
    '''
    if len(X1) < len(X2):
        rand_int = np.random.choice(len(X2), len(X1), replace=False)
        try:
            X2 = X2[rand_int, :]
        except IndexError:
            X2 = X2[rand_int]
    if len(X2) < len(X1):
        rand_int = np.random.choice(len(X1), len(X2), replace=False)
        try:
            X1 = X1[rand_int, :]
        except IndexError:
            X1 = X1[rand_int]
    return X1, X2, rand_int

def Lazy_Balance(num_PF, num_AR):
    '''
    Takes in two numpy arrays and under samples the larger array to have a 1:1 ratio
    '''
    if num_PF < num_AR:

        rand_int = np.random.randint(0, num_AR, num_PF) # sample num_PF samples from num_AR
        upper_limit = num_PF
    else:
        rand_int = np.random.randint(0, num_PF, num_AR) # sample num_AR samples from num_PF
        upper_limit = num_AR

    return rand_int, upper_limit



def interpolate_spectra(spectra_stats_single_obs, raster, lambda_min, lambda_max, field, line, n_breaks, threshold, calib = True):



    hdrs = raster.headers

    lambda_min_mes = hdrs[0]['WAVEMIN']
    lambda_max_mes = hdrs[0]['WAVEMAX']
    delta_lambda = hdrs[0]['CDELT1']
    num_points = hdrs[0]['NAXIS1']
    start_obs = Time(hdrs[0]['STARTOBS'])
    pix_size = hdrs[0]['SUMSPAT']*0.166*u.arcsec
    slit_width = 0.33*u.arcsec

    sol_rad = 696340*u.km

    d = delta_lambda*u.Angstrom/u.pixel # wavelength bin width
    dist = earth_distance(start_obs)
    omega = slit_width*pix_size*(sol_rad/((sol_rad/dist.to(u.km))*u.radian).to(u.arcsec))**2/(1.496E+8*u.km)**2

    n_min = int(np.floor((lambda_min - lambda_min_mes)/delta_lambda))-1
    n_max = int(np.ceil((lambda_max - lambda_min_mes)/delta_lambda))+1 # to make sure the transformation to the desired grid is
                                                                       # possible
    lambda_units = lambda_min_mes + np.arange(0, num_points)*delta_lambda
    lambda_units = lambda_units[n_min:n_max]

    obs_wavelength = np.linspace( lambda_min, lambda_max, num=n_breaks )

    effA = get_interpolated_effective_area(start_obs, response_version=6, detector_type=field,
                                                   obs_wavelength=obs_wavelength*u.Angstrom)
    effA = effA.to(u.cm*u.cm)

    if field == "NUV":
        dn2phot = DN2PHOT['NUV']
        exptime = 'EXPTIMEN'
    else:
        dn2phot = DN2PHOT['FUV']
        exptime = 'EXPTIMEF'

    exp_times = np.asarray([hdrs[n][exptime] for n in range(len(hdrs))])

    raster_aft_exp = raster[:,:,:]/exp_times.reshape(exp_times.shape[0], 1, 1)

    raster_cut = raster_aft_exp[:,:,n_min:n_max]

    print("Original number of spectra : ",
      raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape)

    spectra_stats_single_obs['Original'] = raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape

    plt.show()
    del raster

    #automatic cosmic ray removal, not thoroughly tested so far,
    #readnoise was determined with noisy spectra standard deviation.
    #sigclip and sigfrac was determined with trial and error on one observation in Si IV, detect cosmic is optimized for parallel computing.
    for n in range(raster_cut.shape[0]):

        crmask, _ = astroscrappy.detect_cosmics(raster_cut[n,:,:], sigclip=4.5, sigfrac=0.1, objlim=2, readnoise=2)
        raster_cut[(n,) + np.where(np.any(crmask, axis=-1))] = np.full((1, raster_cut.shape[-1]), fill_value = 0)

    print("remaining spectra after removing cosmic rays : ",
      raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape)

    spectra_stats_single_obs['remaining cosmic'] = raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape

    interpolated_slice = clean_pre_interpol(raster_cut, threshold=threshold, mode='zeros') #, binned1
    print("remaining spectra after cleaning with pre interpol : ",
          interpolated_slice[np.where(np.any(interpolated_slice!=0, axis=-1))].shape)

    spectra_stats_single_obs['cleaned pre interpol'] = interpolated_slice[np.where(np.any(interpolated_slice!=0, axis=-1))].shape

    interpol_f_cut = interp1d(lambda_units, interpolated_slice, kind="linear", axis = -1 )

    obs_wavelength = np.linspace( lambda_min, lambda_max, num=n_breaks )

    interpolated_slice = interpol_f_cut( obs_wavelength )
    interpolated_slice[np.where(np.any(np.isnan(interpolated_slice), axis= -1))] = np.full((1, interpolated_slice.shape[-1]), fill_value = 0)
    calib = True

    if calib:
        # calibrate spectra here
        interpolated_slice_calibrated = ((interpolated_slice/u.second)/((obs_wavelength*u.Angstrom).to(u.cm)*effA*d*omega))*dn2phot*(c.h).to(u.erg*u.second)*(c.c).to(u.cm/u.second)/u.steradian
    else:
        interpolated_slice_calibrated = deepcopy(interpolated_slice)

    interpolated_slice_calibrated, norm_vals = normalize(interpolated_slice_calibrated)

    if line == "Mg II k":
        interpolated_slice_calibrated, binned2 = clean_aft_interpol( interpolated_slice_calibrated.reshape(raster_cut.shape[0]*raster_cut.shape[1], n_breaks), mode = 'zeros' )

        print("remaining spectra after cleaning with aft interpol (only in case of Mg II h & k) : ",
              interpolated_slice_calibrated[np.where(np.any(interpolated_slice_calibrated!=0, axis=-1))].shape)

        spectra_stats_single_obs['clean aft interpol'] = interpolated_slice_calibrated[np.where(np.any(interpolated_slice_calibrated!=0, axis=-1))].shape

    print(raster_cut.shape, interpolated_slice_calibrated.shape)
    interpolated_slice_calibrated = interpolated_slice_calibrated.reshape(raster_cut.shape[:-1] + (n_breaks,))
#     norm_vals = norm_vals.reshape(raster_cut.shape[:-1] + (1,))


    return interpolated_slice_calibrated, norm_vals, spectra_stats_single_obs

#######################################################################################################################

def create_goes(df, obs_id):

    """
    Quick download of Hek and goes obs data
    """

    n = df.index[df['Flare Id'] == obs_id][0]
    start = df.loc[n, 'Start of observation']
    flare_obs_times = df.loc[n, 'Peak time']
    fl_cls = df.loc[n, 'Class']
    if not isinstance(flare_obs_times, list):
        flare_obs_times = [flare_obs_times]

    ###############################################################################
    # obs.goes.events.get_flares(classes="CMX") a lot easier
    tr = a.Time((start-timedelta(hours=1)).isoformat(), (flare_obs_times[-1] + timedelta(hours=1)).isoformat())
    results = Fido.search(tr, a.Instrument.xrs & a.goes.SatelliteNumber(15) | a.hek.FL & (a.hek.FRM.Name == 'SWPC'))  # NOQA

    flare_start = []
    flare_end = []

    ###############################################################################
    # Then download the XRS data and load it into a TimeSeries.

    files = Fido.fetch(results)
    goes = TimeSeries(files, concatenate=True)

    hek_results = results['hek']

    return goes, hek_results


def extract_info(df, obs_id, minutes = 30):

    """
    Extracting info from dataframe about observation with obs_id
    """

    n = df.index[df['Flare Id'] == obs_id][0]
    obs_start = df.at[n, 'Start of observation']
    flare_obs_times = df.at[n, 'Peak time']
    flare_start = df.at[n, 'Start of flare']
    flare_end = df.at[n, 'End of flare']
    fl_other = [other[0] for other in df.at[n, 'All flares']]

    flare_peak = flare_obs_times
    minutesBeforeEvent = [fl - timedelta(minutes=minutes) for fl in df['Start of flare'].loc[n]]


    return obs_start, flare_start, flare_end, flare_peak, minutesBeforeEvent, fl_other


def transform_arrays(times, image_arr=None, norm_vals_arr=None, num_of_raster_pos=0, forward=True):

    """
    transforms between image_arrays in shape (n,t,y,lambda) -> (n*t,y,lambda)
    and time arrays from shape (n,t) -> (n*t,)

    or

    the reverse operation

    """

    if not np.any(image_arr) and forward:

        image_arr = np.zeros(times.shape + (1,)+ (1,))
        norm_vals_arr = np.zeros(times.shape + (1,)+ (1,))

    elif not np.any(image_arr) and not forward:
        image_arr = np.zeros((times.shape[1],) + (1,)+ (1,))

    if forward:

#     if num_of_raster_pos == 1:
#         image_arr_r = image_arr
#         times_r = times
#         norm_vals_r = norm_vals_arr
#     else:
        image_arr_r = image_arr.reshape(image_arr.shape[0]*image_arr.shape[1], image_arr.shape[2], image_arr.shape[3],
                                        order='F')
        times_r = times.reshape(times.shape[0]*times.shape[1], 1, order='F')
        norm_vals_r = norm_vals_arr.reshape(norm_vals_arr.shape[0]*norm_vals_arr.shape[1], norm_vals_arr.shape[2], 1, order='F')

        if np.any(times_r != sorted(times_r)):
            print(times_r[np.where(times_r != sorted(times_r))])
            if num_of_raster_pos == 1:
                sorted_inds = np.squeeze(np.argsort(times_r,axis=0))
#                 print(sorted_inds)
                times_r = times_r[sorted_inds]
                image_arr_r = image_arr_r[sorted_inds,:,:]
                norm_vals_r = norm_vals_r[sorted_inds,:,:]

            else:
                raise Warning('could not reshape times, times are not sequential and number of raster steps is: '
                              , num_of_raster_pos)

        complete_raster_step = times.shape[1]
        raster_pos = np.hstack([np.arange(0,num_of_raster_pos) for n in range(complete_raster_step)])
        times_r = np.vstack([raster_pos,times_r.T])

    else: # reverse

        if num_of_raster_pos > image_arr.shape[0]:

            start_ind = None
            stop_ind = None
            try:
                start_ind = np.where(times[0]==0)[0][0]
            except Exception:
                stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
            times_new = np.zeros([num_of_raster_pos])
            norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

            if start_ind:
                image_arr_new[start_ind:] = image_arr
            elif stop_ind:
                image_arr_new[:stop_ind] = image_arr
            else:
                image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


            print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

            times_origin = deepcopy(times)

            image_arr = image_arr_new
            times = times_new
            norm_vals_arr = norm_vals_new

            image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

            times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

            norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

        elif num_of_raster_pos == 1:

            image_arr_r = image_arr.reshape(1,image_arr.shape[0],image_arr.shape[1],image_arr.shape[2])
            times_r = times[1].reshape(1,times.shape[1])
            norm_vals_r = norm_vals_arr.reshape(1,norm_vals_arr.shape[0],norm_vals_arr.shape[1],1)

        else:


            start_ind = None
            stop_ind = None

            start_ind = np.where(times[0]==0)[0][0]
            stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            if stop_ind < start_ind:

                try:
                    start_ind = np.where(times[0]==0)[0][0]
                except Exception:
                    stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

                image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos])
                norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

                if start_ind:
                    image_arr_new[start_ind:] = image_arr
                elif stop_ind:
                    image_arr_new[:stop_ind] = image_arr
                else:
                    image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


                print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

            else:

                num_of_cycles = int(np.floor((stop_ind - start_ind)/(num_of_raster_pos-1)))


                if start_ind == stop_ind+1:
                    raise ValueError("cannot reshape, start_ind and stop_ind for symmetric reshaping is the same")

                image_arr_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos*(num_of_cycles+2)])
                norm_vals_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), norm_vals_arr.shape[1], 1])

                image_arr_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = image_arr
                times_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = times[1,:]
                norm_vals_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = norm_vals_arr.reshape(norm_vals_arr.shape[0], norm_vals_arr.shape[1],1)

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

#             print(int(times_origin[n][0]))

            # check_diff_1 = [times_r[int(times_origin[0][n]),0] - times_r[int(times_origin[0][0]),0] for n in range(num_of_raster_pos-1)]

#             if times_r.shape[1] > 1:
#                 check_diff_2 = times_r[int(times_origin[0][0]),0] - times_r[int(times_origin[0][0]),1]
#             else:
#                 check_diff_2 = check_diff_1[0]

#             if (np.argmin(check_diff_1) != int(times_origin[0][0])) | np.any(check_diff_2 < check_diff_1[0]):
#                 print(np.argmin(check_diff_1), (check_diff_2 < check_diff_1[0]))
#                 print(f"n={int(times_origin[0][0])}, t=0 :", times_r[int(times_origin[0][0]),0], f"n={int(times_origin[0][0])}+1, t=0 :", times_r[int(times_origin[0][0])+1,0], f"n={int(times_origin[0][0])}, t=1 :", times_r[int(times_origin[0][0]),1])
#                 raise Warning('could not reshape times, times are not in the correct order!')
#             else:
#                 pass

    return times_r, image_arr_r, norm_vals_r


def plotting_goes_curve(goes_truncated, obs_id, flares_events, fl_other, obs_range, save = False):

    """
    Plotting goes curve of observation with 'obs_id'
    """

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    dates = matplotlib.dates.date2num(parse_time(goes_truncated.to_dataframe().index).datetime)

    ax.plot_date(dates, goes_truncated.to_dataframe()['xrsa'], '-',
                  label=r'0.5--4.0 $\AA$', color='blue', lw=2,)
    ax.plot_date(dates, goes_truncated.to_dataframe()['xrsb'], '-',
                  label=r'1.0--8.0 $\AA$', color='violet', lw=2,)
    for event in flares_events:
        ax.axvline(parse_time(event['event_peaktime']).datetime, label = 'Flare')
        ax.axvspan(parse_time(event['event_starttime']).datetime,
                parse_time(event['event_endtime']).datetime,
                alpha=0.3, label = 'Full Flare', color='red')
        ax.axvspan(parse_time(event['halfHourBeforeEvent']).datetime,
                parse_time(event['event_starttime']).datetime,
                alpha=0.3, label = 'Selected time before flare')
    for other in fl_other:
        ax.axvline(parse_time(other).datetime, color='red')
    ax.axvspan(parse_time(obs_range['obs_starttime']).datetime,
                parse_time(obs_range['obs_endtime']).datetime,
                alpha=0.1, label = 'Iris observation time')


    ax.set_yscale('log')

    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-3)
    ax.set_ylabel('Watts m$^{-2}$')

    ax2 = ax.twinx()
    labels = ['A', 'B', 'C', 'M', 'X']
    centers = np.logspace(-7.5, -3.5, len(labels))
    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 1e-3)
    ax2.yaxis.set_minor_locator(mticker.FixedLocator(centers))
    ax2.set_yticklabels(labels, minor=True)
    ax2.set_yticklabels([])
    ax.legend(loc=2)

    ax.yaxis.grid(True, 'major')
    ax.xaxis.grid(False, 'major')
    formatter = matplotlib.dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)

    ax.fmt_xdata = matplotlib.dates.DateFormatter('%H:%M')
    fig.autofmt_xdate()

    filename = f'Plot_goes_{obs_id}'

    plt.show()

    if save == True:
        fig.savefig('sml/zbindenj/First-Paper/plotGOES/' + filename)

def flare_sji_image(sji, vmax, imstep):

    plt.figure( figsize=(10,10))
    plt.imshow( sji.get_image_step(imstep).clip(min=0)**0.4, origin="lower", cmap="gist_heat", vmax=vmax )
    plt.show()

##########################################################################################

def merge(obs_raw1, obs_raw2):
    """
    an object's __dict__ contains all its
    attributes, methods, docstrings, etc.
    """
    obs_raw1.__dict__.update(obs_raw2.__dict__)
    return obs_raw1


def clean_SAA_cls(obs_cls):
        """
        Cleans the parts of observations when IRIS crossed the southern atlantic anomaly SAA and puts them everywhere to zero.

        input:

        im_arr_slit: array containing the spectra ordered either by global time steps or just one slit.

        """
        df_SAA = pd.read_csv('/sml/zbindenj/saa_results.csv')

        times = obs_cls.times_global[1,:]

        for start, end in zip(df_SAA['start'], df_SAA['end']):
            start = datetime.fromisoformat((start.split('.')[0]).split(' ')[-1])
            end = datetime.fromisoformat((end.split('.')[0]).split(' ')[-1])

            start_e = to_epoch(start)
            end_e = to_epoch(end)

            if ((start_e > times[0]) and (start_e < times[-1])) or ((end_e > times[0]) and (end_e < times[-1])):
#                 print(parse_time(start_e, format='unix').to_datetime(), parse_time(end_e, format='unix').to_datetime())
                diff = times - start_e
                try:
                    start_ind = np.argmin(np.abs(diff[diff<0]))
                except Exception:
                    start_ind = 0
#                 print(parse_time(times[start_ind], format='unix').to_datetime())
#                 print(start_ind)
                diff = times - end_e

                try:
                    end_ind = np.argmin(np.abs(diff[diff>0])) + np.argmin(np.abs(diff[diff<0])) + 1 # only take steps outside of SAA
#                     print(parse_time(times[end_ind], format='unix').to_datetime())
#                     print(end_ind)
                except Exception:
                    end_ind = len(times)
#                     print(parse_time(times[end_ind-1], format='unix').to_datetime())
#                     print(end_ind)

                obs_cls.im_arr_global[start_ind:end_ind, :, :] = np.full((obs_cls.im_arr_global[start_ind:end_ind, :, :].shape[0],
                                                                       obs_cls.im_arr_global.shape[1],
                                                                       obs_cls.im_arr_global.shape[2]),
                                                                       fill_value = 0)
                # Times remains unchanged to keep the structure of the arrays intact.
        print("remaining spectra after removing SAA : ",
              obs_cls.im_arr_global[np.where(np.any(obs_cls.im_arr_global!=0, axis=-1))].shape)

        obs_cls.spectra_stats_single_obs['cleaned SAA'] = obs_cls.im_arr_global[np.where(np.any(obs_cls.im_arr_global!=0, axis=-1))].shape


        try:
            delattr(obs_cls, 'times')
            delattr(obs_cls, 'im_arr')
            delattr(obs_cls, 'norm_vals')
        except Exception:
            pass





class Obs_raw_data:

    """

    Class structure to store all the important information for later training and testing of prediction models, train and clean
    with VAE's or overplot SJI images.

        Parameters:
        -------------------
        obs_id : string
            IRIS observation ID
        num_of_raster_pos : int
            Number of raster positions (not times steps)
        times_global : numpy ndarray
            first row contains raster position, second row contains time in unix
        im_arr_global : numpy ndarray
            contains all spectra like (T,y,lambda)
        norm_vals_global : numpy ndarray
            contains the normalization values for each spectrum like (T,y,1)
        n_breaks : int
            interpolation points
        lambda_min : int or float
            lower wavelength limit
        lambda_max : int or float
            upper wavelength limit
        field : string
            FUV or NUV
        line : string
            spectral line
        threshold : int
            lower limit in DN/s at which spectra were cleaned
        hdrs : pandas DataFrame
            containing all headers from raster headers
        threshold : int
            lower limit in DN/s at which spectra were cleaned

    class methods :

        __init__ : initializes a new instance of Obs_raw_data

        time_clipping : clips the observation in time: times_global, im_arr_global, norm_vals_global

            start : datetime
            end : datetime

        save_arr : saves the Obs_raw_data instance according to a specific frame, adjust path in save_arr.

            filename : filename to store the instance as
            line : spectralline
            typ : type of observation : QS, SS, AR, PF

    global methods: check each method for necessary args and kwargs

        clean_SAA_cls(obs_cls) : cleans the given instance for SAA by setting SAA parts to 0

        transform arrays : transforms array between (n,t,y,lambda) <-> (T,y,lambda)
                           CAUTION : timeclipping destroys the equivalence between the two arrays. The function automatically
                           accounts for that by using the first and last complete raster steps.

        spectra_quick_look : allows the user to have a peek at some random spectra

        load_obs_data : allows the user to load a stored Obs_raw_data instance. args/kwargs are the same as in save_arr. Adjust
                        path if necessary


    """

    def __init__(self, obs_id=None, raster=None, lambda_min=None, lambda_max=None, n_breaks=None, field=None, line=None,
                 threshold=None, load_dict=None):

        spectra_stats_single_obs = {}

        if load_dict:

            filename, line, typ = load_dict.values()
            try:
                with h5py.File(f'/sml/zbindenj/{line}/{typ}/{filename}/arrays.h5', 'r') as f:
                    im_arr_global = f["im_arr_global"][:,:,:]
                    times_global = f["times_global"][:,:]
                    norm_vals_global = f["norm_vals_global"][:,:,:]

                init_dict = np.load(f'/sml/zbindenj/{line}/{typ}/{filename}/dict.npz', allow_pickle=True)['arr_0'][()]

                for key, value in init_dict.items():

                    setattr(self, key, value)

                self.im_arr_global = im_arr_global[:,:,:]
                self.times_global = times_global[:,:]
                self.norm_vals_global = norm_vals_global[:,:]

            except Exception as exc:
                print(exc)


        else:

            self.obs_id = obs_id

            self.num_of_raster_pos = raster.n_raster_pos

            times = [raster.get_timestamps(n) for n in range(raster.n_raster_pos)]
            times = np.vstack(times)

            self.lambda_min = lambda_min
            self.lambda_max = lambda_max
            self.n_breaks = n_breaks
            self.field = field
            self.line = line
            self.threshold = threshold
            self.spectra_stats_single_obs = {}

#             interpolated_image_arr = np.zeros([raster.n_raster_pos, raster.get_raster_pos_steps(0), raster.shape[1], n_breaks])
#             norm_val_image_arr = np.zeros([raster.n_raster_pos, raster.get_raster_pos_steps(0), raster.shape[1], 1])

            interpolated_image_clean_norm, norm_val_image, spectra_stats_single_obs = interpolate_spectra(self.spectra_stats_single_obs, raster, lambda_min, lambda_max, field, line, n_breaks, threshold, calib = True)

            self.spectra_stats_single_obs = spectra_stats_single_obs

#             interpolated_image_arr[n_rast,:,:,:] = interpolated_image_clean_norm

#             norm_val_image_arr[n_rast,:,:] = norm_val_image

            self.hdrs = pd.DataFrame(list(raster.headers))
#             self.im_arr = interpolated_image_arr
#             self.norm_vals = norm_val_image_arr


            # transforms image to global raster step
            times_global, _, _ = transform_arrays(times, num_of_raster_pos=raster.n_raster_pos, forward = True)

            self.times_global = times_global # contains raster position data and times for later processing.
            self.im_arr_global = interpolated_image_clean_norm
            self.norm_vals_global = norm_val_image


            #clean out edges
            self.im_arr_global[np.where((np.argmax(self.im_arr_global, axis=-1) > (self.im_arr_global.shape[-1]*0.95)) |
                                        (np.argmax(self.im_arr_global, axis=-1) < (self.im_arr_global.shape[-1]*0.05)))] = 0

            clean_SAA_cls(self)




    def time_clipping(self, start, end):

        """
        Clip image_array and time_array according to start datetime and end datetime, only works with global time steps

        """

        start_e = to_epoch(start)
        end_e = to_epoch(end)

        times = self.times_global[1,:]


        if not (((start_e > times[0]) and (start_e < times[-1])) or ((end_e > times[0]) and (end_e < times[-1]))):
            start_t = parse_time(self.times_global[1,0], format='unix').to_datetime()
            end_t = parse_time(self.times_global[1,-1], format='unix').to_datetime()
            raise Warning(f'in {self.obs_id}: start {start} or end {end} is outside of times: {start_t}, {end_t}')

        start = start_e
        end = end_e

        diff = times - start

        try:
            start_ind = np.argmin(np.abs(diff[diff<0]))
        except ValueError:
            start_ind = 0

        diff = times - end

        end_ind = np.argmin(np.abs(diff[diff<0])) # only take last step before end

        self.times_global = self.times_global[:,start_ind:end_ind] # first dimension contains raster position information
        self.im_arr_global = self.im_arr_global[start_ind:end_ind, :, :] # first dimension contains raster position
                                                                         # information
#         self.im_arr_global_raw = self.im_arr_global_raw[start_ind:end_ind, :, :]
        self.norm_vals_global = self.norm_vals_global[start_ind:end_ind, :]

        # Quick visualization of the selected data ###############################################
#         try:
#             nprof = self.im_arr_global*self.norm_vals_global#.reshape(self.norm_vals_global.shape + (1,))
#         except ValueError:
#             nprof = self.im_arr_global*self.norm_vals_global.reshape(self.norm_vals_global.shape + (1,))


#         nprof = nprof.reshape(nprof.shape[0]*nprof.shape[1], self.n_breaks)

#         nprof = nprof[np.where(~np.all(((nprof == 0) | (nprof == 1)), axis=-1))]

        self.spectra_stats_single_obs['remaining spectra after time clipping'] = self.im_arr_global[np.where(np.any(self.im_arr_global!=0, axis=-1))].shape

#         try:
#             spectra_quick_look(nprof, self.lambda_min, self.lambda_max, self.n_breaks)
#             spectra_quick_look(self.im_arr_global, self.lambda_min, self.lambda_max, self.n_breaks)

#             fig = plt.figure(figsize=(20,20))
#             ax = fig.add_subplot(projection='3d')
#             x = np.arange(nprof.shape[0])
#             y = np.arange(nprof.shape[1])
#             xs, ys = np.meshgrid(x, y)

#             ax.plot_surface(ys.T, xs.T, nprof, cmap=plt.cm.Blues, linewidth=1, alpha=0.9)#, vmin=-5, vmax=+10)

#             ax.axes.set_zlim3d(bottom=-1, top=2000000)
#             ax.view_init(10, -95)
#             plt.show()

#         except Exception as exc:
#             print(exc)
#             print('no data in this time-window')

        ##########################################################################################


    def save_arr(self, filename, line, typ):

        filename = filename.split('.')[0]

        try: # make directory for save file to keep it all together.
            os.mkdir(path=f'/sml/zbindenj/{line}/{typ}/{filename}/')
        except Exception:
            pass


        filename = filename.split('.')[0]

        save_dict = {'obs_id': self.obs_id,
                    'num_of_raster_pos': self.num_of_raster_pos,
                    'lambda_min': self.lambda_min,
                    'lambda_max': self.lambda_max,
                    'n_breaks': self.n_breaks,
                    'field' : self.field,
                    'line' : self.line,
                    'threshold' : self.threshold,
                    'hdrs' : self.hdrs,
                    'stats' : self.spectra_stats_single_obs
                   }

        np.savez(f'/sml/zbindenj/{line}/{typ}/{filename}/dict.npz', save_dict)

        #gc.collect()

        try:

            f = h5py.File(f'/sml/zbindenj/{line}/{typ}/{filename}/arrays.h5', 'w')
            dataset = f.create_dataset("im_arr_global", data=self.im_arr_global)
            dataset = f.create_dataset("times_global", data=self.times_global)
            dataset = f.create_dataset("norm_vals_global", data=self.norm_vals_global)
            f.close()
            del dataset

        except Exception as exc:

            print(exc)
            f.close()


#         da.to_npy_stack(f'/sml/zbindenj/{line}/{typ}/{filename}_im_arr_global.npy', da.from_array(self.im_arr_global, chunks=('auto', 'auto',-1)))
#         da.to_npy_stack(f'/sml/zbindenj/{line}/{typ}/{filename}_times_global.npy', da.from_array(self.times_global, chunks=(-1,'auto')))
#         da.to_npy_stack(f'/sml/zbindenj/{line}/{typ}/{filename}_norm_vals_global.npy', da.from_array(self.norm_vals_global, chunks=('auto', 'auto',-1)))


##########################################################################################

def load_obs_data(filename, line, typ, only_im_arr = False):

    filename = filename.split('.')[0]

    if only_im_arr:

        try:

            f = h5py.File('/fast/zbindenj/{line}/{typ}/{filename}/array.h5', 'r')
            im_arr_global = f["im_arr_global"]
            f.close()
            return im_arr_global

        except Exception as exc:

            print(exc)
            f.close()

            return None

    else:

        load_dict = {
                     'filename' : filename,
                     'line' : line,
                     'typ' : typ
                    }

        return Obs_raw_data(load_dict=load_dict)


def load_obs_data_old(filename, line, typ):
    filename = filename.split('.')[0]
    try:
        obs_data = np.load(f'/sml/zbindenj/{line}/{typ}/{filename}.npz', allow_pickle = True)['arr_0'].item()
    except Exception:
        obs_data = torch.load(f'/sml/zbindenj/{line}/{typ}/{filename}.pt')
    return obs_data

# def load_obs_data_cleaned(filename, line, typ):
#     filename = filename.split('.')[0]

#     if line == 'MgIIk':
#         try:
#             obs_data = np.load(f'/sml/zbindenj/cleaned/{line}/{typ}/{filename}.npz', allow_pickle = True)['arr_0'].item()
#         except Exception:
#             obs_data = torch.load(f'/sml/zbindenj/cleaned/{line}/{typ}/{filename}.pt')
#         return obs_data
#     else:
#         return load_obs_data(filename, line, typ)

class Dataset_VAE(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, labels):
        'Initialization'
        self.labels = labels
        #self.list_IDs = list_IDs
        self.X_data = X #filehandlers

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        #x = self.X_data[ID]
        #y = self.labels[ID]
        x = self.X_data[index]
        y = self.labels[index]
        return x, y

class Dataset_NN(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, label_file_IDs, label_indexs, y_labels, path_to_data):
        'Initialization'
        self.labels = y_labels # 0's and 1's
        self.label_file_IDs = label_file_IDs
        self.label_indexs = label_indexs
        self.path_to_data = path_to_data
#         self.filehandler = {}

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x = f["im_arr_global"][arr_ind]#.clip(min=0)
            y = self.labels[index]
        return x, y

class Dataset_NN_intensity(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, label_file_IDs, label_indexs, y_labels, path_to_data):
        'Initialization'
        self.labels = y_labels # 0's and 1's
        self.label_file_IDs = label_file_IDs
        self.label_indexs = label_indexs
        self.path_to_data = path_to_data
#         self.filehandler = {}

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x = f["im_arr_global"][arr_ind]
            x_i = f["norm_vals_global"][arr_ind]
            X = np.concatenate([x, x_i], axis=-1)
            y = self.labels[index]
        return X, y

##########################################################################################

def norm_log(x, min_, max_):
    """mapping the intensity levels onto a [0,1] in a log shape"""
    x = np.log(x)
    normalized = (x-min_)/(max_-min_)
#     normalized[np.where(np.isnan(normalized) or np.isinf(normalized))] = np.full(normalized.shape, fill_value=0)
    return normalized

##########################################################################################


class SingleLayer(nn.Module):
    def __init__(self, input_dim, d_in2):
        super(SingleLayer, self).__init__()

        self.fc1 = nn.Linear(input_dim, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.dropout(x)

        x = F.relu(self.bc1(self.fc1(x)))

#         x = F.relu(self.fc1(x))

        return F.sigmoid(self.fc2(x))

class TwoLayers(nn.Module):
    def __init__(self, input_dim, d_in2, d_in3):
        super(TwoLayers, self).__init__()

        self.fc1 = nn.Linear(input_dim, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, d_in3)

        self.bc2 = nn.BatchNorm1d(d_in3)

        self.fc3 = nn.Linear(d_in3, 1)

        self.dropout1 = nn.Dropout(0.3)

        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):

        x = self.dropout1(x)

        x = F.relu(self.bc1(self.fc1(x)))

        x = self.dropout2(x)

        x = F.relu(self.bc2(self.fc2(x)))

#         x = F.relu(self.fc1(x))

#         x = F.relu(self.fc2(x))

        return F.sigmoid(self.fc3(x))

class ThreeLayers(nn.Module):
    def __init__(self, input_dim, d_in2, d_in3, d_in4):
        super(ThreeLayers, self).__init__()

        self.fc1 = nn.Linear(input_dim, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, d_in3)

        self.bc2 = nn.BatchNorm1d(d_in3)

        self.fc3 = nn.Linear(d_in3, d_in4)

        self.bc3 = nn.BatchNorm1d(d_in4)

        self.fc4 = nn.Linear(d_in4, 1)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self, x):

        x = self.dropout1(x)

        x = F.relu(self.bc1(self.fc1(x)))

        x = self.dropout2(x)

        x = F.relu(self.bc2(self.fc2(x)))

        x = self.dropout3(x)

        x = F.relu(self.bc3(self.fc3(x)))

#         x = F.relu(self.fc1(x))

#         x = F.relu(self.fc2(x))

#         x = F.relu(self.fc3(x))

        return F.sigmoid(self.fc4(x))


class ConvNet(nn.Module):
    def __init__(self, input_dim, d_in2):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 2, kernel_size=20, stride=4, padding=4),
           nn.ReLU(True))

        self.conv2 = nn.Sequential(
           nn.Conv1d(2, 4, kernel_size=16, stride=3, padding=4),
           nn.ReLU(True))

        self.fc1 = nn.Linear(4*44, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1,1, x.shape[-1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bc1(self.fc1(x)))
#         x = F.relu(self.fc1(x))

#         x = F.relu(self.fc2(x))

        return F.sigmoid(self.fc2(x))

class VGG(nn.Module):
    def __init__(self, input_dim):
        super(VGG, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.conv2 = nn.Sequential(
           nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
           nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.conv4 = nn.Sequential(
           nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
           nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.conv6 = nn.Sequential(
           nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv7 = nn.Sequential(
           nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.conv8 = nn.Sequential(
           nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9 = nn.Sequential(
           nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.conv10 = nn.Sequential(
           nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv11 = nn.Sequential(
           nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.conv12 = nn.Sequential(
           nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),
           nn.ReLU(True))

        self.maxpol6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(1024*10, 100)

        self.bc1 = nn.BatchNorm1d(100)

        self.dropout2 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(100, 1)

#         self.bc2 = nn.BatchNorm1d(1024)

#         self.fc3 = nn.Linear(1024, 128)

#         self.bc3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1,1, x.shape[-1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpol1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpol2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpol3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpol4(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpol5(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpol6(x)

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.bc1(self.fc1(x)))
        x = self.dropout2(x)
#         x = F.relu(self.fc2(x))

#         x = F.relu(self.fc2(x))

        return F.sigmoid(self.fc2(x))





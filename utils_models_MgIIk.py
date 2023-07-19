#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:07:11 2021

@author: jonaszbinden
"""
import sys
if '/home/zbindenj/vae' not in sys.path:
    sys.path.append('/home/zbindenj/vae')
if '/home/zbindenj/features' not in sys.path:
    sys.path.append('/home/zbindenj/features')
import os

from astropy import constants as c


from irisreader import observation, raster_cube
from irisreader.utils.date import to_epoch
from irisreader.utils.date import from_Tformat

import irispy

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
from copy import deepcopy

import autoencoder_MgIIk as autoencoder

import importlib

importlib.reload(autoencoder)


#########################################################################################################################

# path = '/sml/zbindenj/MgIIk/vae/vae_model_7.pt' # MODEL 7

class Dataset_VAE(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, label_file_IDs, label_indexs, path_to_data):
        'Initialization'
        self.label_file_IDs = label_file_IDs
        self.label_indexs = label_indexs
        self.path_to_data = path_to_data
#         self.filehandler = {}
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_indexs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x = deepcopy(f["im_arr_cleaned_QS"][arr_ind])#.clip(min=0)
        return x

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
            x = deepcopy(f["im_arr_cleaned_QS"][arr_ind])#.clip(min=0)
            y = deepcopy(self.labels[index])
        return x, y
    
class Dataset_NN_aggregate(torch.utils.data.Dataset):
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
            x_Mg = deepcopy(f["im_arr_Mg_cleaned_QS"][arr_ind])#.clip(min=0)
            x_Si = deepcopy(f["im_arr_Si_cleaned_QS"][arr_ind])#.clip(min=0)
            x_CII = deepcopy(f["im_arr_CII_cleaned_QS"][arr_ind])#.clip(min=0)
            y = deepcopy(self.labels[index])
        return x_Mg, x_Si, x_CII, y
    
    
    
class Dataset_NN_features(torch.utils.data.Dataset):
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
            x = deepcopy(f["features_cleaned_QS"][arr_ind])
            y = deepcopy(self.labels[index])
        return x, y
    

class SingleLayer(nn.Module):
    def __init__(self, input_dim, d_in2):
        super(SingleLayer, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, d_in2)
        
        self.bc1 = nn.BatchNorm1d(d_in2)
        
        self.fc2 = nn.Linear(d_in2, 1)
        
        self.dropout1 = nn.Dropout(0.5)
        
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):

        x = self.dropout1(x)
        
        x = F.relu(self.bc1(self.fc1(x)))
        
        x = self.dropout2(x)
        
        return F.sigmoid(self.fc2(x))
    
class TwoLayers(nn.Module):
    def __init__(self, input_dim, d_in2, d_in3):
        super(TwoLayers, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, d_in2)
        
        self.bc1 = nn.BatchNorm1d(d_in2)
        
        self.fc2 = nn.Linear(d_in2, d_in3)
        
        self.bc2 = nn.BatchNorm1d(d_in3)
        
        self.fc3 = nn.Linear(d_in3, 1)
        
        self.dropout1 = nn.Dropout(0.5)
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.dropout3 = nn.Dropout(0.5)
        
    def forward(self, x):
        
        x = self.dropout1(x)

        x = F.relu(self.bc1(self.fc1(x)))
        
        x = self.dropout2(x)
        
        x = F.relu(self.bc2(self.fc2(x)))
        
        x = self.dropout3(x)
        
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

   
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, x):

        x = self.dropout3(x)

        x = F.relu(self.bc1(self.fc1(x)))

        x = self.dropout4(x)

        x = F.relu(self.bc2(self.fc2(x)))

        x = self.dropout5(x)

        x = F.relu(self.bc3(self.fc3(x)))

        return F.sigmoid(self.fc4(x))    


    
class SingleLayer_conv(nn.Module):
    def __init__(self, input_dim, d_in2):
        super(SingleLayer_conv, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 10, kernel_size=32, stride=16, padding=4),
           nn.ReLU(True))

        self.conv2 = nn.Sequential(
           nn.Conv1d(10, 20, kernel_size=16, stride=8, padding=4),
           nn.ReLU(True))

        self.conv3 = nn.Sequential(
           nn.Conv1d(20, 40, kernel_size=4, stride=2, padding=2),
           nn.ReLU(True))

        self.fc1 = nn.Linear(40*8, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, 1)

        self.dropout1 = nn.Dropout(0.5)

        self.dropout2 = nn.Dropout(0.5)

        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):

        x = x.view(-1,1, x.shape[-1])

        x = self.conv1(x)

        x = self.dropout1(x)

        x = self.conv2(x)

        x = self.dropout2(x)

        x = self.conv3(x)

        x = torch.flatten(x, 1)

        x = self.dropout3(x)

        x = F.relu(self.bc1(self.fc1(x)))

        return F.sigmoid(self.fc2(x))



class TwoLayers_conv(nn.Module):
    def __init__(self, input_dim, d_in2, d_in3):
        super(TwoLayers_conv, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 10, kernel_size=32, stride=16, padding=4),
           nn.ReLU(True))

        self.conv2 = nn.Sequential(
           nn.Conv1d(10, 20, kernel_size=16, stride=8, padding=4),
           nn.ReLU(True))

        self.conv3 = nn.Sequential(
           nn.Conv1d(20, 40, kernel_size=4, stride=2, padding=2),
           nn.ReLU(True))

        self.fc1 = nn.Linear(40*8, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, d_in3)

        self.bc2 = nn.BatchNorm1d(d_in3)

        self.fc3 = nn.Linear(d_in3, 1)

        self.dropout1 = nn.Dropout(0.5)

        self.dropout2 = nn.Dropout(0.5)

        self.dropout3 = nn.Dropout(0.5)

        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):

        x = x.view(-1,1, x.shape[-1])

        x = self.conv1(x)

        x = self.dropout1(x)

        x = self.conv2(x)

        x = self.dropout2(x)

        x = self.conv3(x)

        x = torch.flatten(x, 1)

        x = self.dropout3(x)

        x = F.relu(self.bc1(self.fc1(x)))

        x = self.dropout4(x)

        x = F.relu(self.bc2(self.fc2(x)))

        return F.sigmoid(self.fc3(x))



class ThreeLayers_conv(nn.Module):
    def __init__(self, input_dim, d_in2, d_in3, d_in4):
        super(ThreeLayers_conv, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 10, kernel_size=32, stride=16, padding=4),
           nn.ReLU(True))

        self.conv2 = nn.Sequential(
           nn.Conv1d(10, 20, kernel_size=16, stride=8, padding=4),
           nn.ReLU(True))

        self.conv3 = nn.Sequential(
           nn.Conv1d(20, 40, kernel_size=4, stride=2, padding=2),
           nn.ReLU(True))

        self.fc1 = nn.Linear(40*8, d_in2)

        self.bc1 = nn.BatchNorm1d(d_in2)

        self.fc2 = nn.Linear(d_in2, d_in3)

        self.bc2 = nn.BatchNorm1d(d_in3)

        self.fc3 = nn.Linear(d_in3, d_in4)

        self.bc3 = nn.BatchNorm1d(d_in4)

        self.fc4 = nn.Linear(d_in4, 1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, x):

        x = x.view(-1,1, x.shape[-1])

        x = self.conv1(x)

        x = self.dropout1(x)

        x = self.conv2(x)

        x = self.dropout2(x)

        x = self.conv3(x)

        x = torch.flatten(x, 1)

        x = self.dropout3(x)

        x = F.relu(self.bc1(self.fc1(x)))

        x = self.dropout4(x)

        x = F.relu(self.bc2(self.fc2(x)))

        x = self.dropout5(x)

        x = F.relu(self.bc3(self.fc3(x)))

        return F.sigmoid(self.fc4(x))    

    
class ConvNet(nn.Module):
    def __init__(self, input_dim, d_in2):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(
           nn.Conv1d(1, 10, kernel_size=32, stride=16, padding=8),
           nn.ReLU(True))
        
        self.conv2 = nn.Sequential(
           nn.Conv1d(10, 20, kernel_size=10, stride=5, padding=5),
           nn.ReLU(True))
        
        self.fc1 = nn.Linear(20*13, d_in2)
        
        self.bc1 = nn.BatchNorm1d(d_in2)
        
        self.fc2 = nn.Linear(d_in2, 1)
        
        self.dropout1 = nn.Dropout(0.5)
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.dropout3 = nn.Dropout(0.5)
        
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1,1, x.shape[-1])
        x = self.dropout1(x)
        
        x = self.conv1(x)
        
        x = self.dropout2(x)
        
        x = self.conv2(x)
        
        x = torch.flatten(x, 1)
        
        x = self.dropout3(x)
        
        x = F.relu(self.bc1(self.fc1(x)))
        
        x = self.dropout4(x)
        
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
        
        self.dropout1 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(1024*10, 100)
        
        self.bc1 = nn.BatchNorm1d(100)
        
        self.dropout2 = nn.Dropout(0.4)

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
    
class VAE_pretrained_convnet(nn.Module):
    def __init__(self, input_dim, latent_dim, d_in2, d_in3):
        super(VAE_pretrained_convnet, self).__init__()
        
        self.ae_encoder = autoencoder.Encoder(latent_dim, 20)
        
        self.ae_decoder = autoencoder.Decoder(latent_dim, 20)
        
        self.fc1 = nn.Linear(latent_dim, d_in2)
        
        self.bc1 = nn.BatchNorm1d(d_in2)
        
        self.fc2 = nn.Linear(d_in2, d_in3)
        
        self.bc2 = nn.BatchNorm1d(d_in3)
        
        self.fc3 = nn.Linear(d_in3, 1)
        
        self.dropout1 = nn.Dropout(.5)
        
        self.dropout2 = nn.Dropout(.5)
       
    
    def forward(self, x):
        x = x.view(-1,1, x.shape[-1])
        
        z, inds1, out_shape1, inds2, out_shape2 = self.ae_encoder(x)
        
        # sample from the distribution having latent parameters z_mu, z_var
#         std = torch.exp(z_var / 2)
#         eps = torch.randn_like(std)
#         # parameterization trick for stochastic gradient descent 
#         z_sample = eps.mul(std).add_(z_mu)

        # decode
        generated_spectra = self.ae_decoder(z, inds1, out_shape1, inds2, out_shape2)
        
        x = self.dropout1(x)
        
        x = F.relu(self.bc1(self.fc1(z)))
        
        x = self.dropout2(x)
        
        x = F.relu(self.bc2(self.fc2(x)))
    
        return generated_spectra, F.sigmoid(self.fc3(x))
    
    
    
    
    
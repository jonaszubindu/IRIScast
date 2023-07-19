#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:07:11 2021

@author: jonaszbinden
"""
import sys
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

class Dataset_VAE(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, label_file_IDs, label_indexs, path_to_data):
        'Initialization'
        self.label_file_IDs = label_file_IDs
        self.label_indexs = label_indexs
        self.path_to_data = path_to_data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_indexs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x = deepcopy(f["im_arr_cleaned_QS"][arr_ind])
        return x

class Dataset_NN(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, label_file_IDs, label_indexs, y_labels, path_to_data):
        'Initialization'
        self.labels = y_labels # 0's and 1's
        self.label_file_IDs = label_file_IDs
        self.label_indexs = label_indexs
        self.path_to_data = path_to_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x = deepcopy(f["im_arr_cleaned_QS"][arr_ind])
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


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_label = self.label_file_IDs[index]
        arr_ind = self.label_indexs[index]
        with h5py.File(self.path_to_data + file_label, 'r') as f:
            x_Mg = deepcopy(f["im_arr_Mg_cleaned_QS"][arr_ind])
            x_Si = deepcopy(f["im_arr_Si_cleaned_QS"][arr_ind])
            x_CII = deepcopy(f["im_arr_CII_cleaned_QS"][arr_ind])
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

        return F.sigmoid(self.fc2(x))

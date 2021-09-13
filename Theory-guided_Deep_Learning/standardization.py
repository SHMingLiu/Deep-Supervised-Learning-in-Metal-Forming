# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:32:09 2020

@author: sl7516
"""

import numpy as np

def standardization(original_data):
    mean = np.mean(original_data)
    std = np.std(original_data)
    data_processed = []
    for i in range(len(original_data)):
        data_processed.append((original_data[i]-mean)/std)
        
    return data_processed, mean, std

def un_standardization(data_processed, mean, std):
    original_data = []
    for i in range(len(data_processed)):
        original_data.append(data_processed[i]*std + mean)
    
    return original_data
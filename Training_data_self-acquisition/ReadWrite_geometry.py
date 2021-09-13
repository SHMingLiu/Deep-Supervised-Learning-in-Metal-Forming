# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:27:37 2020

@author: sl7516
"""
import numpy as np

def read_geometry(file_name):
    fr = open(file_name, 'r')
    sheet_geo_prev = []
    for i, line in enumerate(fr):
        lineArr = line.strip().split()
        if lineArr and lineArr[0].isdigit():
            sheet_geo_prev.append((float(lineArr[1]),float(lineArr[2])))
    sheet_geo_prev = np.array(sheet_geo_prev)
    fr.close()
    return sheet_geo_prev

def write_geometry(file_name, geometry):
    fr = open(file_name, 'w')
    fr.write('Node Label    COORD.COOR1    COORD.COOR2\n')
    for i in range(geometry.shape[0]):
        fr.write('   {}    {}    {}\n'.format(i+1, geometry[i,0], geometry[i,1]))
    fr.close()
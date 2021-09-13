# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:46:05 2020

@author: sl7516
"""
import math
import numpy as np

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def translate(point, vector):
    px, py = point
    qx = px + vector[0]
    qy = py + vector[1]
    return qx, qy

def calibrate_sheet(geometry_input, load_node):
    # import sheet geometry before processing
    sheet_geo_prev = np.array(geometry_input)

    ###### select loading node
    ldnode_id = load_node - 1
    ndnum_top = 1001

    ###### rotation
    # define rotate angle in radians
    rotate_angle = math.atan2((sheet_geo_prev[ldnode_id,1]-sheet_geo_prev[ldnode_id+ndnum_top*12,1]),(sheet_geo_prev[ldnode_id,0]-sheet_geo_prev[ldnode_id+ndnum_top*12,0])) - np.pi/2.0 # radians
    sheet_geo_rotate = np.zeros((sheet_geo_prev.shape[0],sheet_geo_prev.shape[1]))
    for i in range(sheet_geo_prev.shape[0]):
        sheet_geo_rotate[i,0],sheet_geo_rotate[i,1] = rotate(sheet_geo_prev[ldnode_id],sheet_geo_prev[i],-rotate_angle)
    
    ####### translation to origin
    sheet_geo_translate = np.zeros((sheet_geo_prev.shape[0],sheet_geo_prev.shape[1]))
    # define translate vector
    translate_vec = [0-sheet_geo_rotate[ldnode_id+ndnum_top*12,0],0-sheet_geo_rotate[ldnode_id+ndnum_top*12,1]]
    for i in range(sheet_geo_rotate.shape[0]):
        sheet_geo_translate[i,0],sheet_geo_translate[i,1] = translate(sheet_geo_rotate[i],translate_vec)

    ###### offset
    sheet_geo_offset = np.zeros((sheet_geo_prev.shape[0],sheet_geo_prev.shape[1]))
    # define the vertical offset
    offset = [0,abs(min(sheet_geo_translate[:,1]))*10]
    for i in range(sheet_geo_translate.shape[0]):
        sheet_geo_offset[i,0],sheet_geo_offset[i,1] = translate(sheet_geo_translate[i],offset)
    
    return sheet_geo_offset


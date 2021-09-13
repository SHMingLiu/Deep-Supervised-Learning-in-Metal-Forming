# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:33:14 2020

@author: sl7516
"""
import math
import numpy as np

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def write_INP(geometry_inp, stroke):
    ###### select loading stroke
    stroke_inp = stroke + 2

    fr = open('INP_forREAD.inp', 'r')
    fr_modify = open('INP_Name_toBeModified.inp', 'w')

    sheet_geo_offset = np.array(geometry_inp)
    mark_sheet = []
    mark_stroke = []
    for i, line in enumerate(fr):
        lineArr = line.strip().replace(" ", "").split(',')
        if lineArr[0] == '*Part' and lineArr[1] == 'name=Sheet':
            mark_sheet.append(i)
            fr_modify.write('{}'.format(line))
        elif mark_sheet and i > (mark_sheet[0] + 1) and i <= (mark_sheet[0] + 1 + len(sheet_geo_offset)):
            fr_modify.write('      {},   {},   {}\n'.format(int(i-mark_sheet[0]-1),
                                                            round_half_up(sheet_geo_offset[i-mark_sheet[0]-2,0],7),
                                                            round_half_up(sheet_geo_offset[i-mark_sheet[0]-2,1],7)))
        elif lineArr[0] == '**Name:PunchType:Displacement/Rotation': 
            mark_stroke.append(i)
            fr_modify.write('{}'.format(line))
        elif mark_stroke and i == (mark_stroke[0] + 2):
            fr_modify.write('Set-PunchRP, 2, 2, {}\n'.format(float(-stroke_inp)))
        else:
            fr_modify.write('{}'.format(line))

    fr.close()
    fr_modify.close()
    
def modifyINP_iniINC(job_name):    
    fr = open('{}.inp'.format(job_name), 'r')
    fr_modify = open('newIniINC.inp', 'w')
    
    mark_iniINC = []
    for i, line in enumerate(fr):
        lineArr = line.strip().replace(" ", "").split(',')
        if lineArr[0] == '*Step' and lineArr[1] == 'name=Step-Punch':
            mark_iniINC.append(i)
            fr_modify.write('{}'.format(line))
        elif mark_iniINC and i == (mark_iniINC[0] + 2):
            fr_modify.write('{}, 1., 1e-35, 0.01\n'.format(float(lineArr[0]) * 0.1))
        else:
            fr_modify.write('{}'.format(line))
    fr.close()
    fr_modify.close()
    


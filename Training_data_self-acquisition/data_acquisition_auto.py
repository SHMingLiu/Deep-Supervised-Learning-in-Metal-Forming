# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:13:28 2020

@author: sl7516
"""
import numpy as np
import os
import time
import shutil
import matplotlib.pyplot as plt
import json
import traceback

from calibrate_sheet import *
from writeINP_RubberPunch import *
from ReadWrite_geometry import *
from write_MacroCoor import *

def get_newSpace(space,last_space):
    # run one hyper-parameters optimisation trial and save results
    last_ln = last_space['load_node']
    last_stroke = last_space['stroke']
    id_ln = space['load_node'].index(last_ln)
    id_stroke = space['stroke'].index(last_stroke) + 1
    if id_stroke >= len(space['stroke']):
        id_ln += 1
        id_stroke = 0 
    # exclude two option
    if space['load_node'][id_ln] == 101 and space['stroke'][id_stroke] == 17:
        id_ln += 1
        id_stroke = 0 
    if space['load_node'][id_ln] == 901 and space['stroke'][id_stroke] == 17:
        id_ln += 1
        id_stroke = 0 
    new_space = {'load_node': space['load_node'][id_ln],
                      'stroke': space['stroke'][id_stroke]}
    
    return new_space

def read_progress(pwd,space):
    files = os.listdir(pwd)
    for f in files:
        if f.startswith('geometry_'):
            prev = [int(s) for s in f.replace('-', '_').replace('mm','_').split('_') if s.isdigit()]
    prev_punch = int(len(prev)/2)
    preLoad_space = {'prev_load_node': [], 'prev_stroke': []}
    for i in range (prev_punch):
        preLoad_space['prev_load_node'].append(prev[0+i*2])
        preLoad_space['prev_stroke'].append(prev[1+i*2])
    
    if not os.path.exists('results_space.txt.json'):
        new_space = {'load_node': space['load_node'][0],
                          'stroke': space['stroke'][0]} 
        while new_space['load_node'] in preLoad_space['prev_load_node']:
            new_space = get_newSpace(space,new_space)
    else:
        with open('results_space.txt.json', 'r') as f:
            last_space = json.load(f)
        new_space = get_newSpace(space,last_space)
        while new_space['load_node'] in preLoad_space['prev_load_node']:
            new_space = get_newSpace(space,new_space)
    return new_space, prev
    
def exchange_geoFile(pwd,space):
    # exchange the geometry file once the simulations for the last have done
    files = os.listdir(pwd)
    for f in files:
        if f.startswith('geometry_'):
            prev = [int(s) for s in f.replace('-', '_').replace('mm','_').split('_') if s.isdigit()]
            prev_last_punch = {'load_node': prev[-2],'stroke': prev[-1]}
            os.remove(f)
    preLoad_space = {'prev_load_node': []}
    for i in range (int(len(prev)/2)-1):
        preLoad_space['prev_load_node'].append(prev[0+i*2])
        
    updated_prev_lastPunch = get_newSpace(space,prev_last_punch)
    while updated_prev_lastPunch['load_node'] in preLoad_space['prev_load_node']:
        updated_prev_lastPunch = get_newSpace(space,updated_prev_lastPunch)
    # find the new geometry name
    geo_name = 'geometry_'
    for i in range(int(len(prev)/2)-1):
        geo_name = geo_name+str(prev[0+i*2])+'-'+str(prev[1+i*2])+'mm'+'_'
    geo_name = geo_name+str(updated_prev_lastPunch['load_node'])+'-' \
                +str(updated_prev_lastPunch['stroke'])+'mm'+'.txt'
    
    shutil.copy(pwd+'/Previous_geoFile/'+geo_name, pwd)
            
def write_progress(progress):    
    with open('results_space.txt.json', 'w') as f:
        json.dump(
            progress, f,
            default=None, sort_keys=False, 
            indent=4, separators=(',',': ')
            )

def wait_process(name):
    while not os.path.isfile(name):
        time.sleep(1)

def run_sim(space):
    cnt = 0
    pwd = os.getcwd()
    pwd = pwd.replace(os.sep,'/')
    while True:
        try:
            new_space,prev_info = read_progress(pwd,space)
        except Exception as err:
            err_str = str(err)
            print (err_str)
            traceback_str = str(traceback.format_exc())
            print (traceback_str)
            print ('Exchange geometry file\n')
            os.remove('results_space.txt.json')
            exchange_geoFile(pwd,space)
        else:
            print ('Remain geometry file\n')
            break
    load_node,stroke = new_space['load_node'],new_space['stroke']
    pre_name_suffix = ''
    for i in range(int(len(prev_info)/2)):
        pre_name_suffix = pre_name_suffix+str(prev_info[0+i*2])+'-'+str(prev_info[1+i*2])+'mm'
        if i < (int(len(prev_info)/2)-1):
            pre_name_suffix = pre_name_suffix+'_'
    name_suffix = pre_name_suffix+'_'+str(load_node)+'-'+str(stroke)+'mm'
    # read initial geometry
    geomotry_read = read_geometry('geometry_{}.txt'.format(pre_name_suffix))
    # calibrate the sheet to the loading loacation
    geomotry_in = calibrate_sheet(geomotry_read, load_node)
    # write .inp file
    write_INP(geomotry_in, stroke)
    job_name = 'RubberPunch_' + name_suffix
    os.rename('INP_Name_toBeModified.inp', '{}.inp'.format(job_name))  
    while True:
        try:
            # run abaqus simulation
            wait_process('{}.inp'.format(job_name))
            os.system('abaqus job={} cpus=8'.format(job_name))
            wait_process('{}.sta'.format(job_name))
            while True:
                with open('{}.sta'.format(job_name), 'r') as f:
                    line = f.readlines()[-1]
                    last_line = line.strip().split()
                if last_line[0] == 'THE' and last_line[1] == 'ANALYSIS':
                    break
            # extract odb file for deformed geometry
            geometry_name = 'coorFromAbaqus_{}.txt'.format(name_suffix)
            write_MacroCoor(pwd, geometry_name, job_name)
            wait_process('Macro_ExtractCoor.py')
            os.system('abaqus cae noGUI=Macro_ExtractCoor.py')
            wait_process(geometry_name)
            geometry_deformed_read = read_geometry(geometry_name)
        except:
            # modify the initial increment in the .inp if simulation does not converge
            cnt += 1
            os.system('abaqus job={} terminate'.format(job_name))
            os.system('mkdir Trash_{}'.format(cnt))
            Trash_dir = pwd + '/Trash_' + str(cnt)
            modifyINP_iniINC(job_name)
            files = os.listdir(pwd)
            for f in files:
                if f.startswith(job_name):
                    shutil.move(f, Trash_dir)
            wait_process('{}.inp'.format(job_name))
            os.rename('newIniINC.inp', '{}.inp'.format(job_name))  
            print ('Unseccessful simulation at Step-Punch\n')
        else:
            print ('Seccessful simulation at Step-Punch\n')
            break
                
    # calibrate the geometry about the mid
    os.system('abaqus job={} terminate'.format(job_name))
    geometry_deformed = calibrate_sheet(geometry_deformed_read, 501)
    # save the plot of the geometry
    plt.figure(figsize=(8, 6))
    plt.scatter(geometry_deformed[:,0],geometry_deformed[:,1],s=0.1)
    plt.axis('equal')
    plt.savefig('fig_{}.png'.format(name_suffix))
    # write deformed geometry
    file_name_geoDeform = 'geometry_{}.txt'.format(name_suffix)
    write_geometry(file_name_geoDeform, geometry_deformed)
    # write progress
    write_progress(new_space)
    # transfer files
    files = os.listdir(pwd)
    for f in files:
        if f.startswith("geometry_{}".format(name_suffix)):
            shutil.move(f, '{}/Geometry'.format(pwd))
        if f.startswith("fig_"):
            shutil.move(f, '{}/Pics'.format(pwd))
        if f.startswith("RubberPunch"):
            shutil.move(f, '{}/Sims'.format(pwd))
        if f.startswith("coorFromAbaqus_"):
            shutil.move(f, '{}/CoorFromAbaqus'.format(pwd))
        



space = {'load_node': [101, 301, 501, 701, 901],
         'stroke': [1, 5, 9, 13, 17]
         }
os.system('mkdir Geometry')
os.system('mkdir Pics')
os.system('mkdir Sims')
os.system('mkdir CoorFromAbaqus')
trials = 350
for i in range(trials):
    try:
        run_sim(space)
    except Exception as err:
        err_str = str(err)
        print (err_str)
        traceback_str = str(traceback.format_exc())
        print (traceback_str)
        



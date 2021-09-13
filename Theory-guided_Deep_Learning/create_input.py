# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:13:20 2020

@author: sl7516
"""

import numpy as np
#import matplotlib.pyplot as plt


def distance2(x1,x2):
    d = np.sqrt((x1-x2)**2)
    return d

def IDW_calc_2(x,x1,u1,x2,u2):
    w1 = 1/distance2(x,x1)
    w2 = 1/distance2(x,x2)
    u = (w1*u1+w2*u2)/(w1+w2)
    return u

def nearest_2nodesIndex(x,x_all):
    dist = []
    for i in range(len(x_all)):
        dist_temp=distance2(x,x_all[i])
        dist.append(dist_temp)
    dist_sort = dist[:]
    dist_sort.sort()
    index1 = dist.index(dist_sort[0])
    index2 = dist.index(dist_sort[1])
    return index1, index2


def input_generate(stroke, max_depth, trainOrtest):
    Top_node_num = 801
    
    stroke = str(stroke)
    if trainOrtest == "train":
        fr = open('Input_Data/stroke_%(a)s/load_data.txt'%{'a':stroke})  
    if trainOrtest == "test":
        fr = open('Input_Data/test_set/stroke_%(a)s/load_data.txt'%{'a':stroke})  
    x_load=[]
    y_load=[]
    for i, line in enumerate(fr):
        if i > 0 and i < (Top_node_num+1):
            lineArr = line.strip().split() 
            x_load.append(float(lineArr[1]))
            y_load.append(float(lineArr[2]))
    fr.close()
    x_load.reverse()
    y_load.reverse()
    
    x_input = np.linspace(0,80,801)
    y_input = np.ones(801)
    
    ymax = max(y_load)
    
    depth = [ymax-yi for yi in y_load]
    index_0 = 0
    for i in range(len(x_input)):
        if x_input[i] > max(x_load):
            index_0 = i      # the index threshold for parameter interpolation
            break

    for i in range(len(y_input)):
        if i == 0:
            y_input[i] = depth[i]
        elif i > 0 and i < index_0:
            node1,node2 = nearest_2nodesIndex(x_input[i],x_load)
            if x_input[i] == x_load[node1]:
                y_input[i] = depth[node1]
                continue
            y_input[i] = IDW_calc_2(x_input[i],x_load[node1],depth[node1],x_load[node2],depth[node2])
        else:
            for j in range(i,len(x_input)):
                y_input[i] = 0.0
                
    Y_INPUT = [round(y_inputi/max_depth,6) for y_inputi in y_input]

    return Y_INPUT

'''
plt.figure(figsize=(10,6))
plt.scatter(x_input,Y_INPUT,0.3,'b')
plt.scatter(x_load,depth,0.3,'r')
plt.xlim(0,1)


####################
####################   Preprocessing

y_gray = [round(y*(255)/max(y_input),3) for y in y_input]

y_gray = np.reshape(y_gray,(1,801))
gray = y_gray[:]
for i in range(20):
    gray = np.vstack((gray,y_gray))

plt.figure(figsize=(10,6))
plt.imshow(gray, cmap='gray')
plt.show()

plt.figure(figsize=(10,6))
X = np.random.random((100, 100)) # sample 2D array
plt.imshow(X, cmap="gray")
plt.show()
'''
'''
Top_node_num = 801
max_depth = 48.1949
stroke = 19
fr = open('stroke_%(a)s/load_data.txt'%{'a':stroke})  
x_load=[]
y_load=[]
for i, line in enumerate(fr):
    if i > 0 and i < (Top_node_num+1):
        lineArr = line.strip().split() 
        x_load.append(float(lineArr[1]))
        y_load.append(float(lineArr[2]))
fr.close()
x_load.reverse()
y_load.reverse()
    
x_input = np.linspace(0,80,801)
y_input = np.ones(801)
    
ymax = max(y_load)
    
depth = [ymax-yi for yi in y_load]
index_0 = 0
for i in range(len(x_input)):
    if x_input[i] > max(x_load):
        index_0 = i      # the index threshold for parameter interpolation
        break

for i in range(len(y_input)):
    if i == 0:
        y_input[i] = depth[i]
    elif i > 0 and i < index_0:
        node1,node2 = nearest_2nodesIndex(x_input[i],x_load)
        if x_input[i] == x_load[node1]:
            y_input[i] = depth[node1]
            continue
        y_input[i] = IDW_calc_2(x_input[i],x_load[node1],depth[node1],x_load[node2],depth[node2])
    else:
        for j in range(i,len(x_input)):
            y_input[i] = 0.0
                
Y_INPUT = [round(y_inputi/max_depth,3) for y_inputi in y_input]
'''

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:54 2020

@author: sl7516
"""

import tensorflow as tf
from hyperopt import STATUS_OK
import numpy as np
from standardization import *   # for training label standardisation
from auxiliary import load_json_result
import random
import math
import uuid
import os

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

###### change between Reg2 (PIDNN) and noReg (data-driven DNN)
Reg_judge = 'Reg2'

if Reg_judge == 'Reg2':
    MIN_DELTA = 1e-4 # Reg2
    START_STOP_LOSS = 0.5
    from PIDNN_tensorflow_Reg2_lrDecay_spot1 import *
else:
    MIN_DELTA = 1e-9 # noReg
    START_STOP_LOSS = 0.1
    from PIDNN_tensorflow_noReg_lrDecay_spot1 import *

###### load data
BATCH_SIZE = 32
EPOCH = 100000
n_coef = 0.0731 # for AA6082
K_strength = 10 ** 2.6023 # for AA6082
PATIENCE = 5000

WEIGHTS_DIR = 'weights_save/'
TENSORBOARD_DIR = 'TensorBoard/'

###### load training input
total_input = load_json_result('Input_data_4pBend_AA6082.txt.json')
stroke_tot = np.linspace(11.0,31.0,201)
stroke_tot = [round_half_up(i,1) for i in stroke_tot]
# read training and test selections
stroke_train_fr = open('stroke_selection_training_1.txt','r')
stroke_train = np.array([])
for line in stroke_train_fr:
    lineArr = line.strip().split() 
    stroke_train = np.append(stroke_train, float(lineArr[0]))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_test_1.txt','r')
stroke_test = np.array([])
for line in stroke_test_fr:
    lineArr = line.strip().split() 
    stroke_test = np.append(stroke_test, float(lineArr[0]))
stroke_test_fr.close()
'''
# 80% training data = 160, 20% test data = 41
stroke_train = np.random.choice(stroke_tot, 160,replace=False)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_training_3.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_test_3.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 50% training data = 100, 50% test data = 101
stroke_train = np.random.choice(stroke_tot, 100,replace=False)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_50_50_training_3.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_50_50_test_3.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 20% training data = 40, 80% test data = 161
stroke_train = np.random.choice(stroke_tot, 40,replace=False)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_20_80_training_3.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_20_80_test_3.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()

# 10% training data = 20, 90% test data = 181
stroke_train = np.random.choice(stroke_tot, 20,replace=False)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_10_90_training_3.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_10_90_test_3.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()

# 5% training data = 10, 95% test data = 191
stroke_train = np.random.choice(stroke_tot, 10,replace=False)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_5_95_training_3.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_5_95_test_3.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()

# 11-15_30-31 training data = 52, test data = 149
a = stroke_tot[:41]
b = stroke_tot[190:]
stroke_train = np.array(a + b)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_11-15_30-31_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_11-15_30-31_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 11-12_30-31 training data = 22, test data = 179
a = stroke_tot[:11]
b = stroke_tot[190:]
stroke_train = np.array(a + b)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_11-12_30-31_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_11-12_30-31_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 11-12+21-22+30-31 training data = 33, test data = 168
a = stroke_tot[:11]
b = stroke_tot[100:111]
c = stroke_tot[190:]
stroke_train = np.array(a + b + c)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_sym_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_sym_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 11-12+13-14+30-31 training data = 33, test data = 168
a = stroke_tot[:11]
b = stroke_tot[20:31]
c = stroke_tot[190:]
stroke_train = np.array(a + b + c)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_asym_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_asym_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# 11_21_31 training data = 33, test data = 168
stroke_train = np.array(a + b + c)
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_asym_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_asym_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()


# small amount of data 11_13_31 training data = 3, test data = 18
stroke_tot = np.linspace(11.0,31.0,21)
stroke_tot = [round_half_up(i,1) for i in stroke_tot]

stroke_train = np.array([11.0, 21.0, 31.0])
stroke_test = np.setdiff1d(stroke_tot,stroke_train)

# write selections out
stroke_train_fr = open('stroke_selection_sym_smallData_training.txt','w')
for i in stroke_train:
    stroke_train_fr.write('{}\n'.format(i))
stroke_train_fr.close()

stroke_test_fr = open('stroke_selection_sym_smallData_test.txt','w')
for i in stroke_test:
    stroke_test_fr.write('{}\n'.format(i))
stroke_test_fr.close()
'''
Input_ShapeData = []
for i in stroke_train:
    Input_ShapeData.append(total_input['{}'.format(i)])
Input_ShapeData = np.reshape(Input_ShapeData, (-1,801,1)) #input to DNN

test_Input_ShapeData = []
for i in stroke_test:
    test_Input_ShapeData.append(total_input['{}'.format(i)])
test_Input_ShapeData = np.reshape(test_Input_ShapeData, (-1,801,1))


'''
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
for i in range(len(test_Input_ShapeData)):
    if i % 1 ==0:
        plt.plot(test_Input_ShapeData[i,:,0])
'''
total_micro_stdValue = {}
total_stroke_stdValue = {}
total_micro_label = []
fr = open('Training_labels_4pBend_AA6082.txt', 'r')
for i,line in enumerate(fr):
    if i == 0:
        continue
    lineArr = line.strip().split() 
    total_micro_stdValue.update({'{}'.format(lineArr[0]): [0, 0]})
    total_stroke_stdValue.update({'{}'.format(lineArr[0]): 0})
    total_micro_label.append([float(lineArr[1]),float(lineArr[2])])
total_micro_label = np.array(total_micro_label)
# label standardisation
standard_micro_results = []
mean_micro = []
std_micro = []
for i in range(total_micro_label.shape[1]): # standardise the micro-label data
    standard_temp, mean_temp, std_temp = standardization(total_micro_label[:,i])
    standard_micro_results.append(standard_temp)
    mean_micro.append(mean_temp)
    std_micro.append(std_temp)
standard_micro_results = np.transpose(standard_micro_results)
standard_stroke_results, mean_stroke, std_stroke = standardization(stroke_tot)

for i in range(len(standard_micro_results)):
    total_micro_stdValue['{}'.format(stroke_tot[i])]=[standard_micro_results[i,0],standard_micro_results[i,1]]
    total_stroke_stdValue['{}'.format(stroke_tot[i])]=standard_stroke_results[i]

# Micro label & Stroke label for training
Micro_label = []
Stroke_label = []
for i in stroke_train:
    Micro_label.append(total_micro_stdValue['{}'.format(i)])
    Stroke_label.append(total_stroke_stdValue['{}'.format(i)])
Micro_label = np.array(Micro_label) 
Stroke_label = np.reshape(np.array(Stroke_label),(-1,1))

# Micro label & Stroke label for test
test_Micro_label = []
test_Stroke_label = []
for i in stroke_test:
    test_Micro_label.append(total_micro_stdValue['{}'.format(i)])
    test_Stroke_label.append(total_stroke_stdValue['{}'.format(i)])
test_Micro_label = np.array(test_Micro_label)
test_Stroke_label = np.reshape(np.array(test_Stroke_label),(-1,1))

###### funciton for model configuration and training
def build_and_train(hyper_space,log, save_best_weights=True, log_for_tensorboard=False):
    # load model
    if Reg_judge == 'Reg2': # PIDNN
        model = PIDNN_Model(hyper_space, n_coef, K_strength, mean_micro, std_micro)
    else: # data-driven DNN
        model = PIDNN_Model(hyper_space)
    
    model_uuid = str(uuid.uuid4())[:5]
    
    # weights saving 
    if save_best_weights:
        weights_save_path = os.path.join(
                WEIGHTS_DIR, 'lr_{}_lp2_{}_{}/'.format(hyper_space['learn_rate_mult'],hyper_space['reg2_coeff'],model_uuid))
        print ('Model weights will be saved to: {}'.format(weights_save_path),file=log)
        log.flush()
        if not os.path.exists(weights_save_path):
            os.makedirs(weights_save_path)
    
    pred_log = open('{}prediction_log.txt'.format(weights_save_path), 'a')
        
    # TensorBoard log 
    log_path = None
    if log_for_tensorboard:
        log_path = "tensorboard/"
        writer = tf.compat.v1.summary.FileWriter(log_path)
            
    # Training process
    loss = np.zeros(EPOCH)
    test_loss = np.zeros(EPOCH)
    stroke_acc = np.zeros(EPOCH)
    test_stroke_acc = np.zeros(EPOCH)
    
    i_list = np.linspace(0,Input_ShapeData.shape[0]-1,Input_ShapeData.shape[0]) # label the training data from 0 to n
    i_list = [int(ii) for ii in i_list]
    patience_cnt = 0 # used for early stop
    fluctuate_allow = 0 # used for early stop
    fluc_count = 50 # used for early stop
    for epoch in range(EPOCH):
        random.shuffle(i_list)
        full_batch = Input_ShapeData.shape[0] // BATCH_SIZE
        for i in range(full_batch):
            batch_index = i_list[i*BATCH_SIZE:(i*BATCH_SIZE + BATCH_SIZE)]
            batch_input = data_slice(Input_ShapeData,batch_index)
            batch_label_Micro = data_slice(Micro_label,batch_index)
            batch_label_stroke = data_slice(Stroke_label,batch_index)
            
            model.optimizer.run(feed_dict={
                    model.tf_train_InputData : batch_input,
                    model.tf_Micro_labels : batch_label_Micro,
                    model.tf_Stroke_labels : batch_label_stroke
                    })
    
        if Input_ShapeData.shape[0] % BATCH_SIZE != 0:
            batch_index = i_list[(i*BATCH_SIZE + BATCH_SIZE):Input_ShapeData.shape[0]]
            batch_input = data_slice(Input_ShapeData,batch_index)
            batch_label_Micro = data_slice(Micro_label,batch_index)
            batch_label_stroke = data_slice(Stroke_label,batch_index)
            
            model.optimizer.run(feed_dict={
                    model.tf_train_InputData : batch_input,
                    model.tf_Micro_labels : batch_label_Micro,
                    model.tf_Stroke_labels : batch_label_stroke
                    })
    
        # training loss
        epoch_loss = model.loss.eval(feed_dict={
                        model.tf_train_InputData : Input_ShapeData,
                        model.tf_Micro_labels : Micro_label,
                        model.tf_Stroke_labels : Stroke_label
                        })
        # test loss
        epoch_test_loss = model.loss.eval(feed_dict={
                            model.tf_train_InputData : test_Input_ShapeData,
                            model.tf_Micro_labels : test_Micro_label,
                            model.tf_Stroke_labels : test_Stroke_label
                            })
        # training accuracy
        acc_stroke = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(Stroke_label, mean_stroke, std_stroke))
        # test accuracy
        acc_stroke_test = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(test_Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(test_Stroke_label,mean_stroke,std_stroke)) 
        
        # record/update losses/accuracies
        loss[epoch] = float(epoch_loss)
        test_loss[epoch] = float(epoch_test_loss)
        stroke_acc[epoch] = float(acc_stroke)
        test_stroke_acc[epoch] = float(acc_stroke_test)
        
        if (epoch % 200 == 0): # for process monitor
            print ("Epoch loss at epoch {}: {}".format(epoch+1, epoch_loss),file=log)
            print ("training accuracy: stroke({:.2f}%)".format(acc_stroke),file=log)
            print ("test accuracy: stroke({:.2f}%) \n".format(acc_stroke_test),file=log)
            log.flush()
        if (epoch % 2000 == 0): # regular save of model
            model.saver.save(model.session, weights_save_path + 'weights', global_step = epoch+1)
            pred_log.write('epoch {}:\n\n'.format(epoch+1))
            pred_log.write('prediction on training set (stroke, s1, e1, s2, e2, ...):\n\n')
            write_pred(pred_log,model,Input_ShapeData,Micro_label,mean_stroke,std_stroke,mean_micro,std_micro) # add associated new variable in this script
            pred_log.write('prediction on test set (stroke, s1, e1, s2, e2, ...):\n\n')
            write_pred(pred_log,model,test_Input_ShapeData,test_Micro_label,mean_stroke,std_stroke,mean_micro,std_micro)
            write_loss(pred_log,model,epoch_loss,Input_ShapeData,Micro_label,Stroke_label,'training',Reg_judge)
            write_loss(pred_log,model,epoch_test_loss,test_Input_ShapeData,test_Micro_label,test_Stroke_label,'test',Reg_judge)
            write_accuracy(pred_log,acc_stroke,acc_stroke_test)
            pred_log.flush()
        
        if log_for_tensorboard:
            if (epoch % 200 == 0):
                write_op = tf.compat.v1.summary.merge_all()
                merged_summary = write_op.eval(feed_dict={
                        model.tf_train_InputData : batch_input,
                        model.tf_Micro_labels : batch_label_Micro,
                        model.tf_Stroke_labels : batch_label_stroke
                        })
                writer.add_summary(merged_summary, epoch)
        
        
        # early stop
        if loss[epoch] > 1 and epoch > 80000:
            break
        if loss[epoch] < START_STOP_LOSS and stroke_acc[epoch] > 99.9:
            if  epoch > 1 and (loss[epoch-1] - loss[epoch]) > MIN_DELTA:
                if (loss[epoch-2] - loss[epoch-1]) <= MIN_DELTA:
                    fluctuate_allow += 1
                patience_cnt = 0
            else:
                patience_cnt += 1
                if fluctuate_allow > 2:
                    fluc_count -= 1
                if fluc_count < 1:
                    print ('Early Stopping on flucuation count...',file=log)
                    log.flush()
                    break
        
            if patience_cnt > PATIENCE:
                print ('Early Stopping ...',file=log)
                log.flush()
                break
    
    ###### training complete
    # model save
    model.saver.save(model.session, weights_save_path  + 'weights', global_step = epoch+1)
    # write results out
    pred_final_results = open('{}prediction_final_results.txt'.format(weights_save_path), 'a')
    pred_final_results.write('epoch {}:\n\n'.format(epoch+1))
    pred_final_results.write('prediction on training set (stroke, s1, e1, s2, e2, ...):\n\n')
    write_pred(pred_final_results,model,Input_ShapeData,Micro_label,mean_stroke,std_stroke,mean_micro,std_micro) # add associated new variable in this script
    pred_final_results.write('prediction on test set (stroke, s1, e1, s2, e2, ...):\n\n')
    write_pred(pred_final_results,model,test_Input_ShapeData,test_Micro_label,mean_stroke,std_stroke,mean_micro,std_micro)
    write_loss(pred_final_results,model,epoch_loss,Input_ShapeData,Micro_label,Stroke_label,'training',Reg_judge)
    write_loss(pred_final_results,model,epoch_test_loss,test_Input_ShapeData,test_Micro_label,test_Stroke_label,'test',Reg_judge)
    write_accuracy(pred_final_results,acc_stroke,acc_stroke_test)
    pred_final_results.close()
    
    # evaluate the test loss after training
    loss_test = model.loss.eval(feed_dict={
                            model.tf_train_InputData : test_Input_ShapeData,
                            model.tf_Micro_labels : test_Micro_label,
                            model.tf_Stroke_labels : test_Stroke_label
                            })
    # test accuracy
    stroke_acc_test = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(test_Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(test_Stroke_label,mean_stroke,std_stroke))
    test_loss = test_loss[:epoch+1].tolist()
    test_stroke_acc = test_stroke_acc[:epoch+1].tolist()
    loss = loss[:epoch+1].tolist()
    stroke_acc = stroke_acc[:epoch+1].tolist()
            
    model_name = 'model_trainAcc_{}_testAcc{}_{}'.format(str(round(stroke_acc[-1],9)),str(round(stroke_acc_test,3)),str(uuid.uuid4())[:5])
    print ('Model name: {}'.format(model_name),file=log)
    print ('stroke_acc_test: {}'.format(stroke_acc_test),file=log)
    log.flush()
    
    # record data
    history = {'test_loss': test_loss,
               'test_stroke_acc': test_stroke_acc,
               'loss': loss,
               'stroke_acc': stroke_acc
              }
    result = {
            'loss': float(-stroke_acc_test),
            'last_learn_rate': float(model.current_learn_rate.eval()),
            'real_testloss': float(loss_test),
            'stroke_best_test_accuracy': float(max(history['test_stroke_acc'])),
            'stroke_best_test_loss': float(min(history['test_loss'])),
            'stroke_end_test_accuracy': float(stroke_acc_test),
            'stroke_end_test_loss': float(loss_test),
            'model_name': model_name,
            'hyper_space': hyper_space,
            'status': STATUS_OK,
            'history': history}
    pred_log.close()
    
    return model, model_name, result, log_path
    

def accuracy(predictions, labels):
    diff = []
    for i in range(len(predictions)):
        diff.append(np.abs(predictions[i]-labels[i]))
    accuracy = []
    for i in range(len(diff)):
        accuracy.append(100.0 * (1 - np.abs(diff[i]/labels[i])))
    return (np.sum(accuracy)/len(accuracy))

def data_slice(total_data,index_list):
    new_data = total_data[index_list[0]]
    new_data = np.expand_dims(new_data,0)
    for i in range(1,len(index_list)):
        batch_temp = total_data[index_list[i]]
        new_data = np.vstack((new_data,np.expand_dims(batch_temp,0)))
    return new_data

def write_pred(file,model,Input,Mic_label,mean_stroke,std_stroke,mean_micro,std_micro):
    stroke_p = un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(Input,(-1,801,1))}),mean_stroke,std_stroke)
    pred = stroke_p
    for i in range(Mic_label.shape[1]):
        micro_temp = un_standardization(model.micro_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(Input,(-1,801,1))})[:,i],mean_micro[i],std_micro[i])
        micro_temp = [[ele] for ele in micro_temp]
        pred = np.column_stack((pred, micro_temp))
    col_width = max(len(str(word)) for row in pred for word in row) + 2
    for row in pred:
        file.write(''.join(str(word).ljust(col_width) for word in row))
        file.write('\n')
    file.write('\n')
    
def write_loss(file,model,total_loss,Input,label_micro,label_stroke,train_test,reg_judge):
    stroke_loss = model.loss_Stroke.eval(feed_dict={
                                model.tf_train_InputData : Input,
                                model.tf_Micro_labels : label_micro,
                                model.tf_Stroke_labels : label_stroke
                                })
    micro_loss = model.loss_Micro.eval(feed_dict={
                                model.tf_train_InputData : Input,
                                model.tf_Micro_labels : label_micro,
                                model.tf_Stroke_labels : label_stroke
                                })
    if reg_judge == 'Reg2':
        Reg2_loss = model.loss_Reg2.eval(feed_dict={
                                            model.tf_train_InputData : Input_ShapeData,
                                            model.tf_Micro_labels : Micro_label,
                                            model.tf_Stroke_labels : Stroke_label
                                            })
        file.write('loss composition on {} set (stroke, micro, Reg2, total):\n\n'.format(train_test))
        file.write('{}  {}  {}  {}\n\n'.format(stroke_loss,micro_loss,Reg2_loss,total_loss))
    else:
        file.write('loss composition on {} set (stroke, micro, total):\n\n'.format(train_test))
        file.write('{}  {}  {}\n\n'.format(stroke_loss,micro_loss,total_loss))
    
def write_accuracy(file,accTrain,accTest):
    file.write('stroke accuracy (training, test):\n\n{} {}\n\n'.format(accTrain, accTest))
    file.write('######################################################################\n\n')



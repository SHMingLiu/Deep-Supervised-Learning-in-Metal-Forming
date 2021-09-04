# -*- coding: utf-8 -*-
"""
Neural network training.
@author: Shiming Liu
Using: tensorflow 2.0
"""
import tensorflow as tf
import numpy as np
from create_input import *      # for input nomalisation
from standardization import *   # for training label standardisation
import random

import uuid
import os


###### change between Reg2 (PIDNN) and noReg (data-driven DNN)
Reg_judge = 'Reg2'

if Reg_judge == 'Reg2':
    MIN_DELTA = 1e-4 # Reg2
    START_STOP_LOSS = 0.5
    from PIDNN_tensorflow_Reg2_lrDecay import *
else:
    MIN_DELTA = 1e-9 # noReg
    START_STOP_LOSS = 0.1
    from PIDNN_tensorflow_noReg_lrDecay import *



###### load data
NUM_IN = 3
BATCH_SIZE = 2
EPOCH = 200000
n_coef = 0.0731 # for AA6082
K_strength = 10 ** 2.6023 # for AA6082
F_hill = 2/np.sqrt(3)
PATIENCE = 5000
max_depth = 23.9803 # 4-point bending AA6082
#max_depth = 18.14333 # air bending AA6082
#max_depth = 35.2676 # 4-point bending SS400


WEIGHTS_DIR = 'weights_save/'
TENSORBOARD_DIR = 'TensorBoard/'

###### select number of inputs
if NUM_IN == 11:   # full training data
    a = 2
    b = 1

if NUM_IN == 6:    # half training data
    a = 4
    b = 2

if NUM_IN == 3:    # 1/4 training data
    a = 10
    b = 5

if NUM_IN == 2:
    a = 20
    b = 10


shape_input = []
for i in range(11,33,a):
    shape_input.append(input_generate(i, max_depth, "train"))  

Input_ShapeData = np.reshape(shape_input, (-1,801,1)) #input to DNN

###### load training label
def lessData_select(originial_data):
    data = []
    for i in range(0,len(originial_data),b):
        data.append(originial_data[i])
    return data

Micro_label = np.load('Micro_label_MidPoint_11trainData.npy')
Stroke_label = np.load('Stroke_label_MidPoint_11trainData.npy')

Micro_exp = []
for i in range(len(np.transpose(Micro_label))): #
    micro_l = Micro_label[:,i]
    micro_l = lessData_select(micro_l)
    Micro_exp.append(micro_l)

s_theta_R0_standard, mean_s_theta_R0, std_s_theta_R0 = standardization(Micro_exp[0])
e_theta_R0_standard, mean_e_theta_R0, std_e_theta_R0 = standardization(Micro_exp[2])

Micro_label = np.transpose([s_theta_R0_standard,Micro_exp[1],e_theta_R0_standard]) # Micro label

stroke = lessData_select(Stroke_label)
stroke_standard, mean_stroke, std_stroke = standardization(stroke)
Stroke_label = np.array(stroke_standard) # Stroke label


def test_standard(original_data, mean, std):
    data_processed = []
    for i in range(len(original_data)):
        data_processed.append((original_data[i]-mean)/std)
    return data_processed

test_shape_input = []
for i in range(12,32,2):
    test_shape_input.append(input_generate(i, max_depth, "test"))  


test_Input_ShapeData = np.reshape(test_shape_input, (-1,801,1)) #input to DNN

test_Micro_label = np.load('test_Micro_label_MidPoint_11trainData.npy')
test_Stroke_label = np.load('test_Stroke_label_MidPoint_11trainData.npy')
test_s_theta_R0_standard = test_standard(test_Micro_label[:,0],mean_s_theta_R0,std_s_theta_R0)
test_e_theta_R0_standard = test_standard(test_Micro_label[:,2],mean_e_theta_R0,std_e_theta_R0)
test_Micro_label = np.transpose([test_s_theta_R0_standard,test_Micro_label[:,1],test_e_theta_R0_standard]) # Micro label

test_Stroke_label = np.array(test_standard(test_Stroke_label,mean_stroke,std_stroke))

def build_and_train(hyper_space, save_best_weights=True, log_for_tensorboard=False):
    if Reg_judge == 'Reg2':
        model = PIDNN_Model(hyper_space,n_coef,K_strength,F_hill,mean_s_theta_R0,std_s_theta_R0,mean_e_theta_R0,std_e_theta_R0)
    else:
        model = PIDNN_Model(hyper_space)
    
    model_uuid = str(uuid.uuid4())[:5]
    
    
    # weights saving 
    if save_best_weights:
        weights_save_path = os.path.join(
                WEIGHTS_DIR, 'lr_{}_lp2_{}_{}/'.format(hyper_space['learn_rate_mult'],hyper_space['reg2_coeff'],model_uuid))
        print ('Model weights will be saved to: {}'.format(weights_save_path))
        if not os.path.exists(weights_save_path):
            os.makedirs(weights_save_path)
        
        
    # TensorBoard log 
    log_path = None
    if log_for_tensorboard:
        log_path = "tensorboard/"
        writer = tf.compat.v1.summary.FileWriter(log_path)
            
    # Train net
    loss = np.zeros(EPOCH)
    val_loss = np.zeros(EPOCH)
    test_loss = np.zeros(EPOCH)
    stroke_acc = np.zeros(EPOCH)
    val_stroke_acc = np.zeros(EPOCH)
    test_stroke_acc = np.zeros(EPOCH)
    end_val_stroke_acc = []
    
    i_list = np.linspace(0,Input_ShapeData.shape[0]-1,Input_ShapeData.shape[0])
    i_list = [int(ii) for ii in i_list]
    patience_cnt = 0
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
    
        # !! cross validation used only for 3 input data
        if Input_ShapeData.shape[0] % BATCH_SIZE != 0:
            batch_index = i_list[(i*BATCH_SIZE + BATCH_SIZE):Input_ShapeData.shape[0]]
            batch_index = i_list[(i*BATCH_SIZE + BATCH_SIZE):Input_ShapeData.shape[0]]
            batch_input = data_slice(Input_ShapeData,batch_index)
            batch_label_Micro = data_slice(Micro_label,batch_index)
            batch_label_stroke = data_slice(Stroke_label,batch_index)
            
            # loss for cross-validation data
            epoch_val_loss = model.loss.eval(feed_dict={
                            model.tf_train_InputData : batch_input,
                            model.tf_Micro_labels : batch_label_Micro,
                            model.tf_Stroke_labels : batch_label_stroke
                            })
            epoch_acc_stroke_val = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: batch_input}),mean_stroke,std_stroke),un_standardization(batch_label_stroke,mean_stroke,std_stroke)) 
    
        epoch_loss = model.loss.eval(feed_dict={
                        model.tf_train_InputData : Input_ShapeData,
                        model.tf_Micro_labels : Micro_label,
                        model.tf_Stroke_labels : Stroke_label
                        })
        epoch_test_loss = model.loss.eval(feed_dict={
                            model.tf_train_InputData : test_Input_ShapeData,
                            model.tf_Micro_labels : test_Micro_label,
                            model.tf_Stroke_labels : test_Stroke_label
                            })
        acc_stroke = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(Stroke_label, mean_stroke, std_stroke))
        acc_stroke_test = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(test_Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(test_Stroke_label,mean_stroke,std_stroke)) 
        
        loss[epoch] = float(epoch_loss)
        val_loss[epoch] = float(epoch_val_loss)
        test_loss[epoch] = float(epoch_test_loss)
        stroke_acc[epoch] = float(acc_stroke)
        val_stroke_acc[epoch] = float(epoch_acc_stroke_val)
        test_stroke_acc[epoch] = float(acc_stroke_test)
        
        if (epoch % 200 == 0):
            print ("Epoch loss at epoch {}: {}".format(epoch+1, epoch_loss))
            print ("training accuracy: stroke({:.2f}%)".format(acc_stroke))
            print ("validation accuracy: stroke({:.2f}%)".format(epoch_acc_stroke_val))
            print ("test accuracy: stroke({:.2f}%) \n".format(acc_stroke_test))
        if (epoch % 2000 == 0):
            model.saver.save(model.session, weights_save_path + 'weights', global_step = epoch+1)
        
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
                patience_cnt = 0
                end_val_stroke_acc=[]
            else:
                patience_cnt += 1
                end_val_stroke_acc.append(float(epoch_acc_stroke_val))
        
            if patience_cnt > PATIENCE:
                print ('Early Stopping ...')
                break
    
    # test net
    model.saver.save(model.session, weights_save_path  + 'weights', global_step = epoch+1)
    loss_test = model.loss.eval(feed_dict={
                            model.tf_train_InputData : test_Input_ShapeData,
                            model.tf_Micro_labels : test_Micro_label,
                            model.tf_Stroke_labels : test_Stroke_label
                            })
    stroke_acc_test = accuracy(un_standardization(model.stroke_predict.eval(feed_dict={model.tf_train_InputData: np.reshape(test_Input_ShapeData,(-1,801,1))}),mean_stroke,std_stroke),un_standardization(test_Stroke_label,mean_stroke,std_stroke))
    val_loss = val_loss[:epoch+1].tolist()
    test_loss = test_loss[:epoch+1].tolist()
    val_stroke_acc = val_stroke_acc[:epoch+1].tolist()
    test_stroke_acc = test_stroke_acc[:epoch+1].tolist()
    loss = loss[:epoch+1].tolist()
    stroke_acc = stroke_acc[:epoch+1].tolist()
    if end_val_stroke_acc:
        end_val_stroke_acc = np.mean(end_val_stroke_acc)
    else:
        end_val_stroke_acc = 0
    
    if len(val_stroke_acc) == EPOCH:
        end_val_stroke_acc = np.mean(val_stroke_acc[-5000:])
    model_name = 'model_valAcc_{}_testAcc{}_{}'.format(str(round(end_val_stroke_acc,9)),str(round(stroke_acc_test,3)),str(uuid.uuid4())[:5])
    print ('Model name: {}'.format(model_name))
    print ('stroke_acc_test: {}'.format(stroke_acc_test))
    
    history = {'val_loss': val_loss,
               'test_loss': test_loss,
               'val_stroke_acc': val_stroke_acc,
               'test_stroke_acc': test_stroke_acc,
               'loss': loss,
               'stroke_acc': stroke_acc
              }
    result = {
            'loss': float(-end_val_stroke_acc),
            'last_learn_rate': float(model.current_learn_rate.eval()),
            'real_testloss': float(loss_test),
            'stroke_best_val_accuracy': float(max(history['val_stroke_acc'])),
            'stroke_best_val_loss': float(min(history['val_loss'])),
            'stroke_best_test_accuracy': float(max(history['test_stroke_acc'])),
            'stroke_best_test_loss': float(min(history['test_loss'])),
            'stroke_end_test_accuracy': float(stroke_acc_test),
            'stroke_end_test_loss': float(loss_test),
            'model_name': model_name,
            'hyper_space': hyper_space,
            'status': STATUS_OK,
            'history': history}
    
    
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

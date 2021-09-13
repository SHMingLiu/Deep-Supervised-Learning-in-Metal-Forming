# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:22:15 2020

@author: sl7516

using tensorflow 2.0
"""

import tensorflow as tf 
import numpy as np 
tf.compat.v1.disable_eager_execution()



class PIDNN_Model:
    
    def __init__(self, hyper_space, n_coef, K_strength, mean_micro, std_micro, restore=False):
        self.start_learn_rate = hyper_space['learn_rate_mult']
        self.l_p2 = hyper_space['reg2_coeff']       
        self.n_coef,self.K_strength = n_coef,K_strength
        self.mean_s_von_1,self.mean_e_eq_1 = mean_micro[0],mean_micro[1]
        self.std_s_von_1,self.std_e_eq_1 = std_micro[0],std_micro[1]
        self.tf_train_InputData,self.micro_predict,self.stroke_predict,self.s_von_1,self.e_eq_1,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_conv4,self.b_conv4,self.W_conv5,self.b_conv5,self.W_conv6,self.b_conv6,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2,self.W_fc3,self.b_fc3,self.W_fc4,self.b_fc4 = self.Define_DNN()
        
        self.Training_Method()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
        self.session = tf.compat.v1.InteractiveSession()
        self.session.run(tf.compat.v1.global_variables_initializer())
        
        if restore:
            checkpoint = tf.train.get_checkpoint_state("weights_save")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print ('PIDNN training successfully restored from', checkpoint.model_checkpoint_path)
            else:
                print ('No previous PIDNN model restored, start a new training')
        else:
            print ('No previous PIDNN model restored, start a new training')
            
    def Weight_Var(self, shape, Name):
        w_vars = tf.random.truncated_normal(shape, stddev=0.01)
        return tf.Variable(w_vars, name=Name)
    
    def Bias_Var(self, shape, Name):
        bias_vars = tf.constant(0.01, shape=shape)
        return tf.Variable(bias_vars, name=Name)
    
    def conv1d(self, x, W, stride):
        return tf.nn.conv1d(x, W, stride = stride, padding = "VALID")
    
    def Define_DNN(self):
        tf_train_InputData = tf.compat.v1.placeholder(tf.float32,shape=[None, 801, 1], name="input_trainData")
        
        W_conv1 = self.Weight_Var([3,1,16], "W_conv1")
        b_conv1 = self.Bias_Var([16], "b_conv1")
    
        W_conv2 = self.Weight_Var([3,16,32], "W_conv2")
        b_conv2 = self.Bias_Var([32], "b_conv2")
    
        W_conv3 = self.Weight_Var([3,32,32], "W_conv3")
        b_conv3 = self.Bias_Var([32], "b_conv3")
    
        W_conv4 = self.Weight_Var([3,32,64], "W_conv4")
        b_conv4 = self.Bias_Var([64], "b_conv4")
    
        W_conv5 = self.Weight_Var([3,64,64], "W_conv5")
        b_conv5 = self.Bias_Var([64], "b_conv5")
    
        W_conv6 = self.Weight_Var([3,64,64], "W_conv6")
        b_conv6 = self.Bias_Var([64], "b_conv6")
    
        W_fc1 = self.Weight_Var([6080,1000], "W_fc1") 
        b_fc1 = self.Bias_Var([1000], "b_fc1")
        
        W_fc2 = self.Weight_Var([1000,100], "W_fc2") 
        b_fc2 = self.Bias_Var([100], "b_fc2")
    
        W_fc3 = self.Weight_Var([100,10], "W_fc3") 
        b_fc3 = self.Bias_Var([10], "b_fc3")
        
        W_fc4 = self.Weight_Var([10,3], "W_fc4") 
        b_fc4 = self.Bias_Var([3], "b_fc4")
    
    
        with tf.name_scope("conv1"):
            h_conv1 = tf.nn.relu(self.conv1d(tf_train_InputData,W_conv1,1) + b_conv1)
            h_pool_1 = tf.compat.v1.layers.max_pooling1d(h_conv1, pool_size=2, strides=2, padding='VALID')
            tf.compat.v1.summary.histogram("weights", W_conv1)
            tf.compat.v1.summary.histogram("biases", b_conv1)
            tf.compat.v1.summary.histogram("activations", h_conv1)
        
        with tf.name_scope("conv2"):
            h_conv2 = tf.nn.relu(self.conv1d(h_pool_1,W_conv2,1) + b_conv2)
            tf.compat.v1.summary.histogram("weights", W_conv2)
            tf.compat.v1.summary.histogram("biases", b_conv2)
            tf.compat.v1.summary.histogram("activations", h_conv2)
        with tf.name_scope("conv3"):
            h_conv3 = tf.nn.relu(self.conv1d(h_conv2,W_conv3,1) + b_conv3)
            h_pool_2 = tf.compat.v1.layers.max_pooling1d(h_conv3, pool_size=2, strides=2, padding='VALID')
            tf.compat.v1.summary.histogram("weights", W_conv3)
            tf.compat.v1.summary.histogram("biases", b_conv3)
            tf.compat.v1.summary.histogram("activations", h_conv3)
        
        with tf.name_scope("conv4"):
            h_conv4 = tf.nn.relu(self.conv1d(h_pool_2,W_conv4,1) + b_conv4)
            tf.compat.v1.summary.histogram("weights", W_conv4)
            tf.compat.v1.summary.histogram("biases", b_conv4)
            tf.compat.v1.summary.histogram("activations", h_conv4)
        with tf.name_scope("conv5"):
            h_conv5 = tf.nn.relu(self.conv1d(h_conv4,W_conv5,1) + b_conv5)
        with tf.name_scope("conv6"):
            h_conv6 = tf.nn.relu(self.conv1d(h_conv5,W_conv6,1) + b_conv6)
            h_pool_3 = tf.compat.v1.layers.max_pooling1d(h_conv6, pool_size=2, strides=2, padding='VALID')
            tf.compat.v1.summary.histogram("weights", W_conv6)
            tf.compat.v1.summary.histogram("biases", b_conv6)
            tf.compat.v1.summary.histogram("activations", h_conv6)
        
        print ("dimension:",h_pool_3.get_shape().as_list())
        h_pool_3_flat = tf.reshape(h_pool_3,[-1,6080]) # 95x64 = 6080
        print ("Flat_dimension:",h_pool_3_flat.get_shape().as_list())
        
        with tf.name_scope("fc1"):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_3_flat,W_fc1) + b_fc1)
            tf.compat.v1.summary.histogram("weights", W_fc1)
            tf.compat.v1.summary.histogram("biases", b_fc1)
            tf.compat.v1.summary.histogram("activations", h_fc1)
        
        with tf.name_scope("fc2"):
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
            tf.compat.v1.summary.histogram("weights", W_fc2)
            tf.compat.v1.summary.histogram("biases", b_fc2)
            tf.compat.v1.summary.histogram("activations", h_fc2)
        
        with tf.name_scope("fc3"):
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2,W_fc3) + b_fc3)
            tf.compat.v1.summary.histogram("weights", W_fc3)
            tf.compat.v1.summary.histogram("biases", b_fc3)
            tf.compat.v1.summary.histogram("activations", h_fc3)

        with tf.name_scope("fc4"):
            h_fc4 = tf.matmul(h_fc3,W_fc4) + b_fc4
            tf.compat.v1.summary.histogram("weights", W_fc4)
            tf.compat.v1.summary.histogram("biases", b_fc4)
            
            micro_predict = tf.slice(h_fc4, [0,0], [-1,2])
            stroke_predict = tf.slice(h_fc4, [0,2], [-1,1])
    
            s_von_1 = tf.slice(micro_predict, [0,0], [-1,1])*self.std_s_von_1 + self.mean_s_von_1
            e_eq_1 = tf.slice(micro_predict, [0,1], [-1,1])*self.std_e_eq_1 + self.mean_e_eq_1
    
        return tf_train_InputData,micro_predict,stroke_predict,s_von_1,e_eq_1,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_conv4,b_conv4,W_conv5,b_conv5,W_conv6,b_conv6,W_fc1,b_fc1,W_fc2,b_fc2,W_fc3,b_fc3,W_fc4,b_fc4
    
    def Training_Method(self):
        self.tf_Micro_labels = tf.compat.v1.placeholder(tf.float32,shape=[None, 2], name="input_Micro_Label")
        self.tf_Stroke_labels = tf.compat.v1.placeholder(tf.float32,shape=[None, 1], name="input_Stroke_Label")
        
        with tf.name_scope("loss"):
            self.loss_Micro = tf.math.reduce_mean(tf.math.square(self.tf_Micro_labels - self.micro_predict))
            tf.compat.v1.summary.scalar("Loss_Micro", self.loss_Micro)
            self.loss_Stroke = tf.math.reduce_mean(tf.math.square(self.tf_Stroke_labels - self.stroke_predict))
            tf.compat.v1.summary.scalar("Loss_Stroke", self.loss_Stroke)
            self.loss_Reg2 = tf.math.reduce_mean(self.l_p2 * tf.math.square(self.s_von_1-self.K_strength*tf.math.pow(tf.math.abs(self.e_eq_1)+0.000000001,self.n_coef)))
            tf.compat.v1.summary.scalar("Loss_Reg2", self.loss_Reg2)
    
            self.loss = self.loss_Micro + self.loss_Stroke + self.loss_Reg2
        
        self.global_step = tf.Variable(0)
        self.current_learn_rate = tf.compat.v1.train.exponential_decay(self.start_learn_rate, self.global_step, 100000, 0.96, staircase=True)
        
        with tf.name_scope("optimizer"):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.current_learn_rate).minimize(self.loss, global_step=self.global_step)
        
        
        
        

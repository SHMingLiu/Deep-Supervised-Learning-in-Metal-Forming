# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:14:24 2020

@author: sl7516

using tensorflow 2.0
"""

from trainDNN_tensorflow import build_and_train
from auxiliary import print_json, save_json_result, load_best_hyperspace
from hyperopt import STATUS_FAIL, hp, tpe, fmin, Trials
import tensorflow as tf

import json
import numpy as np
import traceback
import os

# hyperopt space
space = {
        # learn rate from 1e-8 - 1e-5
        'learn_rate_mult': [1e-3,1e-4,1e-5,1e-6],
        'reg2_coeff': [0.1,0.001,0.0001]
        }

def optimise_DNN(hyper_space,log):
    # optimise the hyper-parameters of DNN
    try:
        model, model_name, result, _ = build_and_train(hyper_space,log)
        
        # save training results
        save_json_result(model_name, result)
        with open('results_hyperspace.txt.json', 'w') as f:
            json.dump(
                result['hyper_space'], f,
                default=None, sort_keys=False, 
                indent=4, separators=(',',': ')
                )
        
        model.session.close()
        del model
        
        #return result
    
    except Exception as err:
        try:
            model.session.close()
        except:
            pass
        err_str = str(err)
        print (err_str,file=log)
        traceback_str = str(traceback.format_exc())
        print (traceback_str,file=log)
        return {
                'status': STATUS_FAIL,
                'err': err_str,
                'traceback': traceback_str
                }
    print ('\n\n',file=log)
    log.flush()
        

def get_newHyper(hyper_space,last_hyperspace):
    # run one hyper-parameters optimisation trial and save results
    if (space.keys()) == 2:
        last_lr = last_hyperspace['learn_rate_mult']
        last_lp2 = last_hyperspace['reg2_coeff']
        id_lr = hyper_space['learn_rate_mult'].index(last_lr) + 1
        id_lp2 = hyper_space['reg2_coeff'].index(last_lp2)
        if id_lr >= len(hyper_space['learn_rate_mult']):
            id_lr = 0
            id_lp2 += 1
        new_hyperspace = {'learn_rate_mult': hyper_space['learn_rate_mult'][id_lr],
                          'reg2_coeff': hyper_space['reg2_coeff'][id_lp2]}
    else:
        last_lr = last_hyperspace['learn_rate_mult']
        id_lr = hyper_space['learn_rate_mult'].index(last_lr) + 1
        new_hyperspace = {'learn_rate_mult': hyper_space['learn_rate_mult'][id_lr]}
    
    return new_hyperspace

def run_a_trial(log):
    if not os.path.exists('results_hyperspace.txt.json'):
        if len(space.keys()) == 2:
            new_hyperspace = {'learn_rate_mult': space['learn_rate_mult'][0],
                              'reg2_coeff': space['reg2_coeff'][0]}  
        else:
            new_hyperspace = {'learn_rate_mult': space['learn_rate_mult'][0]} 
    else:
        with open('results_hyperspace.txt.json', 'r') as f:
            last_hyperspace = json.load(f)
        new_hyperspace = get_newHyper(space,last_hyperspace)
        
    optimise_DNN(new_hyperspace,log)
    
    print ('\nOPTIMISATION STEP COMPLETE.\n',file=log)
    log.flush()
    
    
############################################################## Run the whole optimisation process
optimise_trials = 1
    
log = open('log_file.txt', 'a')
for i in range(optimise_trials):
    try:
        run_a_trial(log)
    except Exception as err:
        err_str = str(err)
        print (err_str,file=log)
        traceback_str = str(traceback.format_exc())
        print (traceback_str,file=log)
        log.flush()
log.close()

    
    


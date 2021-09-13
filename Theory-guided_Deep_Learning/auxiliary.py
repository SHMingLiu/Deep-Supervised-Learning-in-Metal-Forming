# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:37:12 2020

@author: sl7516
"""

from bson import json_util
import json
import os

RESULTS_DIR = 'results/'


def print_json(result):
    # print result in a jsonable structure
    print (json.dumps(
            result,
            default=None, sort_keys=False, #json_util.default
            indent=4, separators=(',',': ')
            ))

def save_json_result(model_name, result):
    # save the result in json format
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
                result, f,
                default=None, sort_keys=False, #json_util.default
                indent=4, separators=(',',': ')
                )
        
def load_json_result(best_result_name):
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.load(f)
        
def load_best_hyperspace():
    results = [
            f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
            ]
    if len(results) ==0:
        return None
    
    best_result_name = results[-1]
    return load_json_result(best_result_name)['hyper_space']
    
    
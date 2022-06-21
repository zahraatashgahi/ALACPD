
"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
from sklearn import preprocessing
import numpy as np
import argparse

import os
import json
from load_dataset import TimeSeries
import tensorflow as tf
import datetime 

"""*************************************************************************"""
"""                           General Functions                              """                
import shutil       
import errno
def remove_dir(dir):
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
def check_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise      
                
                
"""*************************************************************************"""
"""                        Parameters Initialization                        """                
def init_parameters():        
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('--dataset_name', type=str, default='har', help='dataset name')
    parser.add_argument("--seed", default=0, help="seed", type=int)
    parser.add_argument("--windows", default=6, help="windows", type=int)
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='AE_skipLSTM_AR', help='model_name')
    parser.add_argument("--unit", default=20, help="unit", type=int)
    parser.add_argument("--horizon", default=4, help="horizon", type=int)
    parser.add_argument("--highway", default=6, help="highway", type=int)
    parser.add_argument("--skip_sizes", nargs="+", default=[3, 5, 7], type=int)

    # Train options
    parser.add_argument("--epochs", default=10, help="epochs", type=int)    
    parser.add_argument("--train_percent", type=float, default=0.1, help="train_percent")
    args = parser.parse_args()
    
    params = {'dataset_name': args.dataset_name,
                "skip_sizes":args.skip_sizes,
                'train_percent':args.train_percent, 
                "training_epochs":args.epochs, 
                "horizon" : args.horizon, 
                "highway":args.highway,
                "model_names": [args.model_name , args.model_name , args.model_name ],
                "lr": 0.001, "GRUUnits":args.unit, "SkipGRUUnits":args.unit, 
                "window": args.windows,
                "num_change_threshold": 7,
                "num_ano_cpd": 3,
                "threshold_high": 4, "threshold_low": 1.4,
                "epochs_to_train_after_cpd":100,
                "extra_samples_after_cpd":3,
                "epochs_to_train_single_sample": 5}
   
    # Initialize model path
    params["file_name"] = "./results_"+params["model_names"][0]+\
                    "/"+args.dataset_name+"/seed="+str(args.seed)+"/"
    check_path(params["file_name"])
    params["ensemble_space"] = len(params["model_names"])
    
    return args, params

def load_empty_model_assests(ensemble_space):
    cpdnet_init = [None] * ensemble_space
    Data = [None] * ensemble_space
    cpdnet = [None] * ensemble_space
    cpdnet_tensorboard = [None] * ensemble_space
    graph = [None] * ensemble_space 
    sess = [None] * ensemble_space
    return cpdnet_init, Data, cpdnet, cpdnet_tensorboard, graph, sess



"""*************************************************************************"""
"""                                  Read Data                              """    
def load_data(args):
    dataset_name =  args.dataset_name
    if dataset_name in ['occupancy', 'apple', 'bee_waggle_6', 'run_log']:
        #data = load_data(dataset_name)
        ts = TimeSeries.from_json("./datasets/"+ dataset_name+"/"+dataset_name +".json")
        data = ts.y
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        np.savetxt("./data/"+dataset_name+".txt", data,fmt='%.6f',  delimiter=',')
        time = ts.datestr
        with open('./datasets/annotations/annotations.json') as json_file:
            annotations = json.load(json_file)    
            annotations = annotations[dataset_name]
    return data, annotations, time
    
    







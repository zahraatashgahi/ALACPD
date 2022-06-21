"""************************************************************************"""
"""**********                     Imports                          ********"""  

import os
import os; import sys; sys.path.append(os.getcwd())

import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from shutil import copyfile
from tensorflow.keras.models import model_from_json
import pickle

from cpdnet_util import CPDNetInit, SetArguments
from utils import load_empty_model_assests, check_path
from cpdnet_datautil import DataUtil
from lstmae_ensemble import AE_skipLSTM_AR, AE_skipLSTM,  AR, ModelCompile


"""************************************************************************"""
"""**********                    New Model                         ********"""  

def model(args, params, data):
    cpdnet_init, Data, cpdnet, cpdnet_tensorboard, graph, sess = load_empty_model_assests(params["ensemble_space"])
    skip_sizes =  params["skip_sizes"]
    
    for j in range(params["ensemble_space"]):
        skip = skip_sizes[j]
        args2 = SetArguments(data = "data/"+args.dataset_name+".txt", filename=params["file_name"],
                            save = "model" ,
                            epochs = params["training_epochs"], skip=skip, CNNKernel=0,
                            window=params["window"], batchsize=1 , 
                            horizon = params["horizon"], highway=params["highway"],
                            lr = params["lr"], GRUUnits=params["GRUUnits"],
                            SkipGRUUnits=params["SkipGRUUnits"],
                            debuglevel=50,optimizer = "SGD", normalize = 0,
                            trainpercent=params['train_percent'],
                            validpercent=0, no_validation=False,
                            tensorboard="", predict="all", plot=True)


        ###--------------- Initialise parameters
        cpdnet_init[j] = CPDNetInit(args2, args_is_dictionary=True)
        ###--------------- Offilne Training ------------------------------
        cpdnet[j], cpdnet_tensorboard[j],  Data[j], graph[j], sess[j] =\
                 offline_training(params["model_names"][j], j, cpdnet_init[j])

       
    return cpdnet_init, Data, cpdnet, cpdnet_tensorboard, graph, sess



def offline_training(model_nme, j, cpdnet_init):     
    print("Python version: %s", sys.version)
    print("Tensorflow version: %s", tf.__version__)
    print("Keras version: %s ... Using tensorflow embedded keras", tf.keras.__version__)

    # Dumping configuration
    cpdnet_init.dump()

    # Reading data
    Data = DataUtil(cpdnet_init.data,
                    cpdnet_init.trainpercent, cpdnet_init.validpercent,
                    cpdnet_init.horizon, cpdnet_init.window,
                    cpdnet_init.normalise)
    if Data.train[0].shape[0] == 0:
        print("Training samples are low\n\n\n\n\n\n\n\n\n")
        exit(0)   
    print("Training shape: X:%s Y:%s", str(Data.train[0].shape), str(Data.train[1].shape))
    print("Validation shape: X:%s Y:%s", str(Data.valid[0].shape), str(Data.valid[1].shape))
    print("Testing shape: X:%s Y:%s", str(Data.test[0].shape), str(Data.test[1].shape))

    if cpdnet_init.plot == True and cpdnet_init.autocorrelation is not None:
        AutoCorrelationPlot(Data[j], cpdnet_init)
    

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        with sess.as_default():
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", flush=True)
            print("^^^^^^^^^^^ Creating model")
            if model_nme =="AE_skipLSTM_AR":
                cpdnet = AE_skipLSTM_AR(cpdnet_init, Data.train[0].shape)
            elif model_nme== "AE_skipLSTM":
                cpdnet = AE_skipLSTM(cpdnet_init, Data.train[0].shape)
            elif model_nme == "AR":
                cpdnet = AR(cpdnet_init, Data.train[0].shape)
            
            if cpdnet is None:
                print("Model could not be loaded or created ... exiting!!")
                exit(1)
            
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", flush=True)
            print("^^^^^^^^^^^ Compiling the model")
            sess.run(tf.global_variables_initializer())
            # Compile model
            cpdnet_tensorboard = ModelCompile(cpdnet, cpdnet_init)
            print("Model compiled")

            # Model Training
            if cpdnet_init.train is True:
                # Train the model
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", flush=True)
                print("Training model on normal data... ")
                h = train(cpdnet, Data, cpdnet_init, cpdnet_tensorboard)


    return cpdnet, cpdnet_tensorboard, Data, graph, sess


def train(model, data, init, tensorboard = None):
    tensorboard = None
    if init.validate == True:
        val_data = (data.valid[0], data.valid[1])
    else:
        val_data = None

    start_time = datetime.now()
    history = model.fit(
                x = data.train[0],
                y = data.train[1],
                epochs = init.epochs,
                batch_size = init.batchsize,
                validation_data = (data.train[0], data.train[1]),
                callbacks = [tensorboard]  if tensorboard else None
            )
    end_time = datetime.now()
    print("Training time: %s", str(end_time - start_time))

    return history


import numpy as np
def eval_data(ensemble_space, cpdnet, graph, sess, x, y):
    loss = [None] * ensemble_space 
    for j in range(ensemble_space):
        with graph[j].as_default():
            with sess[j].as_default():
                loss[j] = cpdnet[j].evaluate(x,y, verbose=0,)      
    return np.asarray(loss)




def train2(x, y, ensemble_space, cpdnet, graph, sess, cpdnet_init, tensorboard = None, epochs = 50):
    tensorboard = None
    for j in range(ensemble_space):
        with graph[j].as_default():
            with sess[j].as_default():
                cpdnet[j].fit(
                    x, y,verbose=0, epochs= epochs,
                    batch_size = cpdnet_init[j].batchsize,
                    validation_data = (x, y) , callbacks = [tensorboard[j]]  if tensorboard else None
                )
    return cpdnet, graph, sess 



"""************************************************************************"""
"""**********                     Imports                          ********"""    
import numpy as np
import json
import os; import sys; sys.path.append(os.getcwd())
sys.path.insert(0, './code/')
import sys; sys.path.append(os.getcwd())
import tensorflow as tf
from sklearn import preprocessing

""" my packages """
from plt_utils import plot_results
from utils import *
from model_util import model, eval_data, train2
import pickle
from lstmae_ensemble import  ModelCompile
from metrics import f_measure, covering    
from cpdnet_datautil import DataUtil
   
    
    
   
    
    
"""************************************************************************"""
"""**********                 Initialization                       ********"""    
#----------------------------------------------- Read arguments
args, params = init_parameters()
print(args)
print(params)

#-----------------------------------------------  Set Random seed
import os
os.environ['PYTHONHASHSEED']=str(args.seed)
import random
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)
import tensorflow as tf
tf.random.set_random_seed(args.seed)       

#----------------------------------------------- Read data    
data, annotations, time = load_data(args)

print("Log path ", params["file_name"] +"log.out")
# redirect print
sys.stdout = open(params["file_name"] +"log.out", 'w')
   
"""************************************************************************"""
"""**********                 Generate Models                      ********""" 
print("#########################################################################################")
print("#########                   Generate Models and Offline Training                 ########")
print("#########################################################################################")
cpdnet_init, Data, cpdnet, cpdnet_tensorboard, graph, sess = model(args, params, data)

"""*************************************************************************"""
"""**********              Calculate Loss- normal data              ********""" 
print("#########################################################################################")
print("#########                   Calcualate Mean Loss on Normal Data                  ########")
print("#########################################################################################")
loss_normal = eval_data(params["ensemble_space"], cpdnet, graph, sess,
                        Data[0].train[0], Data[0].train[1])               
print("Loss Normal Data = ", loss_normal)   



    
"""************************************************************************"""
"""**********                 Online Training                      ********""" 

print("#########################################################################################")
print("#########                             Online Training                            ########")
print("#########################################################################################")
# num samples
m_train = Data[0].train[0].shape[0]
m_test  = Data[0].test[0].shape[0]
m_normal = m_train

# Initialize results lists
ano_indices_train = []
ano_indices_plot  = []
cpd_indices = []
y_ano = (m_train+m_test) * [0]
data_loss = np.zeros((data.shape[0], params["ensemble_space"]))
data_mean_loss = np.zeros((data.shape[0], params["ensemble_space"]))
for i in range(cpdnet_init[0].window*2 + (m_normal-1)):
    data_loss[i,:] = loss_normal
    data_mean_loss[i,:] = loss_normal
counter_ano = 0
num_ano_cpd = params["num_ano_cpd"]
th1 = params["threshold_high"]
th2 = params["threshold_low"]

#-------------------------------------------------  Save results   


# ------------------------------------------------




# iterate over all test data
for i in range(m_test): 
    print("-------------------------------------------------------------------------", flush=True)
    #--------------------------- new data to test ----------------------------
    idx = cpdnet_init[0].window*2 + (m_train+i-1)
    x_in = np.expand_dims(Data[0].test[0][i], axis=0)
    y_in = np.expand_dims(Data[0].test[1][i], axis=0)
    #--------------------------- evaluate the loss-----------------------------
    l_test_i = eval_data(params["ensemble_space"], cpdnet, graph, sess, x_in, y_in)  
    data_loss[idx,:] = l_test_i
    
        
    print("X_test ", idx, "--> mean loss= ", np.mean(l_test_i), "    loss:", l_test_i)
    # If loss is larger than a threshold then the sample is anomaly else it is normal
    if m_normal > params["num_change_threshold"]: 
        th = th2 
    else: 
        th = th1
    #---------------------------- Check loss ---------------------------------
    if np.sum([l_test_i[k]>= th * loss_normal[k] for k in range(params["ensemble_space"])])> (params["ensemble_space"]/2) \
        and i > 8 and m_normal>4:
        #if np.mean(l_test_i) > np.mean(loss_normal) * th:
        print("#### ^^^^^^^^^^^^^^^^^^    --> Anomaly Detected ^^^^^^^^^^^^^^^^ ####", flush=True)
        y_ano[m_train + i] = 1
        counter_ano += 1
        ano_indices_train.append(i)
        ano_indices_plot.append(idx)
        
        if counter_ano > num_ano_cpd:
            print("#########################################################################################")
            print("#########                           Change-point Detected                        ########")
            print("#########################################################################################", flush=True)
            # insert cpd index and remove the wrongly detected anomalies from the list
            cpd_indices.append(idx - num_ano_cpd + int(cpdnet_init[0].window/2))
            print(cpd_indices)
            print("cpd point: ", idx - num_ano_cpd + int(cpdnet_init[0].window/2))
            print("train on :", ano_indices_train)
            print("train on (plot) :", ano_indices_plot)
            for i in range(num_ano_cpd+1):
                del ano_indices_plot[-(num_ano_cpd-i)]
            
            counter_ano = 0
            # train model with the new samples
            x = Data[0].test[0][ano_indices_train[-1]:ano_indices_train[-1]+params["extra_samples_after_cpd"]]
            y = Data[0].test[1][ano_indices_train[-1]:ano_indices_train[-1]+params["extra_samples_after_cpd"]]
            m_normal = 0
            cpdnet, graph, sess= train2( x, y,  params["ensemble_space"], cpdnet, graph,
                                        sess, cpdnet_init, cpdnet_tensorboard, 
                                        epochs = params["epochs_to_train_after_cpd"])
            loss_normal = eval_data(params["ensemble_space"], cpdnet, graph, sess, x, y)               
            print("new loss = ", loss_normal, flush=True)   
            

            ano_indices_train = []
            plot_results(data, time, data_loss, data_mean_loss, ano_indices_plot, cpd_indices,
                      save_dir=cpdnet_init[0].logfilename, name=args.dataset_name,
                      current_idx =  idx)
            print("#########################################################################################")
            print("#########                    Analysing Change-point Is Finished                  ########")
            print("#########                        Model has been Adapted                          ########")
            print("#########################################################################################")
        data_mean_loss[idx,:] = loss_normal
  
    else:
        counter_ano = 0
        ano_indices_train = []
        m_normal += 1
        if m_normal > params["extra_samples_after_cpd"]:
            cpdnet, graph, sess= train2( x_in, y_in,  params["ensemble_space"], cpdnet, graph,
                                            sess, cpdnet_init, cpdnet_tensorboard, 
                                            epochs= params["epochs_to_train_single_sample"])
            l_test_i_n = eval_data(params["ensemble_space"], cpdnet, graph, sess,  x_in, y_in) 
            # update mean loss of normal data
            loss_normal = (loss_normal*(m_normal-1)+ l_test_i_n)/(m_normal)
            print("new average loss of normal data = ", loss_normal, flush=True)
        else:
            #print("pass")
            cpdnet, graph, sess= train2( x_in, y_in,  params["ensemble_space"], cpdnet, graph,
                                            sess, cpdnet_init, cpdnet_tensorboard, 
                                            epochs = params["epochs_to_train_single_sample"])
        
            l_test_i_n = eval_data(params["ensemble_space"], cpdnet, graph, sess,  x_in, y_in) 
            loss_normal = (loss_normal*(m_normal-1)+ l_test_i_n)/(m_normal)
            print("new average loss of normal data = ", loss_normal, flush=True)
        data_mean_loss[idx,:] = loss_normal
     
    plot_results(data, time, data_loss, data_mean_loss, 
              ano_indices_plot, cpd_indices, 
              save_dir=cpdnet_init[0].logfilename, name=args.dataset_name, 
              current_idx = idx)
    np.savetxt(params["file_name"]  + 'cpd_indices.out', np.asarray(cpd_indices), delimiter=',')




print("#########################################################################################")
print("#########                            Algorithm is Finished                       ########")
print("#########################################################################################")
        
print("predicted = ", cpd_indices)
if args.dataset_name in ['occupancy', 'apple', 'bee_waggle_6', 'run_log']:
    file1 = open(params["file_name"] + "results.txt","w") 
    print("\nreal = ", annotations)
    file1.write("\npredicted = " + str(cpd_indices)) 
    file1.write("\nreal = " + str(annotations)) 
    
    print("\ncovering: ", covering( annotations, cpd_indices,  data.shape[0]))
    print("\nf_measure: ", f_measure( annotations, cpd_indices))    
    file1.write("\ncovering = " + str(covering( annotations, cpd_indices,  data.shape[0]))) 
    file1.write("\nf_measure = " + str(f_measure( annotations, cpd_indices))) 
    
    file1.close()







    
    



# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:51:21 2020

@author: 20194461
"""
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pyplot import text
import pandas as pd
from load_dataset import TimeSeries
    
def plot_results(x, time, loss, mean_loss, ano_indices, cpd_indices, current_idx, save_dir='./', name="dataset"):
    m = x.shape[1]
    n = x.shape[0]
    ensemble_space = loss.shape[1]

    from datetime import datetime
    time2 = []
    for t in time:
        if name == "occupancy":
            time2.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
        elif name == "apple":
            time2.append(datetime.strptime(t, "%Y-%m-%d"))
        elif name == "run_log":
            time2.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
        elif name == "bee_waggle_6":
            time2.append(t)
    
    current_t =  np.argwhere(loss[:,0] == 0)[0]
    
    if name == "occupancy":
        labels = ['temperature', 'relative humidity', 'light', 'CO2']
        xl = "Time"
    elif name == "apple":
        labels = ['daily closing price', 'volume']
        xl = "Time"
    elif name == "run_log":
        labels = ['pace', 'total distance traveled']
        xl = "Time"
    elif name == "bee_waggle_6":
        labels = ['x position', 'y position', 'sine of the head angle', 'cosine of the head angle']
        xl = "Move index"
        
    ts = TimeSeries.from_json("./datasets/"+ name+"/"+name +".json")
    data = ts.y    
    
        
    """------------------      plot results only       --------------------"""
    num_graphs = m
    ts = pd.DataFrame(data, columns=labels, index= time2)  
    axes = ts.plot(subplots=True, legend=True, figsize=(20,num_graphs*3), colormap='copper')
   
    # plot change-points
    for k in cpd_indices:
        for j in range(num_graphs):
            axes[j].axvline(time[k], color='red',  alpha=0.8) 
    
    for j in range(num_graphs):
        axes[j].legend(loc = "upper right")
        axes[j].axvline(time[current_t], color='green',  alpha=0.8) 
        #if j < (ensemble_space-1):
        #    axes[j].set_xticks([])

    if name == "run_log":
        x_labels = ts.index.strftime("%Y-%m-%d %H:%M:%S")
        axes[-1].set_xticklabels(x_labels)
    
    plt.xlabel(xl)
    plt.savefig(save_dir+"_result_"+name+".pdf", bbox_inches='tight')
    plt.show()   
    
    
 
    #------------------      plot results and loss       --------------------
    num_graphs = ensemble_space
    labels = []
    for i in range(num_graphs):
        labels.append("Model " + str(i+1))
    
    fig, axs = plt.subplots(ensemble_space, 1, figsize=[20,num_graphs*3], sharey=False, sharex=False)
    axs = axs.flatten()
    
    for i in range(ensemble_space):
        axs[i].plot(time2, loss[:,i], label = "Loss", color='tab:red')
        axs[i].plot(time2, mean_loss[:,i], label = "Average Loss",  color='tab:blue')
        axs[i].set_ylabel("Loss (TAENet "+str(i+1)+")")
        if i < (ensemble_space-1):
            axs[i].set_xticks([])
        axes[j].axvline(time[current_t], color='green',  alpha=0.8) 
    #axs[-1].set_xticks(time2)
    axs[-1].legend(loc = "upper right")
    
    if name == "run_log":
        x_labels = ts.index.strftime("%Y-%m-%d %H:%M:%S")
        axs[-1].set_xticklabels(x_labels)
    plt.xlabel(xl)
    plt.savefig(save_dir+"_loss_"+name+".pdf", bbox_inches='tight')
    plt.show()   


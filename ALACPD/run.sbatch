#!/bin/bash

set -e
set -x

model_name="AE_skipLSTM_AR"  #AE_skipLSTM AR
seed=0 
windows=6
train_percent=0.1 
dataset="run_log" #occupancy  apple   bee_waggle_6 
				
python3 ./code/main.py  --dataset_name $dataset\
						--seed $seed\
						--windows $windows\
						--train_percent $train_percent\
						--model_name $model_name\
						--epochs 10



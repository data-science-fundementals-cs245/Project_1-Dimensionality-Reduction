#!/bin/bash

rtype=LDA


# paras for model

dims=100
aim_acc=0.7
max_dim=100
method='importances'
step = 5

~/Anaconda3/python.exe scripts/main.py --rtype $rtype --dims $dims --aim_acc $aim_acc --max_dim $max_dim --method $method --step $step

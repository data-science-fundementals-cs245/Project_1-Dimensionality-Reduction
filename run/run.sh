#!/bin/bash

rtype=LDA


# paras for model

dims=100
aim_acc=0.7
max_dim=100

/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 scripts/main.py --rtype $rtype --dims $dims --aim_acc $aim_acc --max_dim $max_dim

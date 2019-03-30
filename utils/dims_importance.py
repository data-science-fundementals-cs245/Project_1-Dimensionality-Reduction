"""
Author: Hongbin Chen

Use ExtraTree decision model to classify the training set and obtain feature importances for each feature

File 'feature_importances.pkl' stores the the importances for each feature
File 'feature_importances_order.pkl' stores the ascent order of feature importances
"""
#-*- coding:utf-8 -*-
import os, sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import pickle as pkl

root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from utils.data_reader import Data


if __name__ == "__main__":
    data_set = Data()
    train_X, test_X, train_y, test_y = data_set.Get_Train_Test() 
    print("Loading data finished!")
    clf = ExtraTreesClassifier()
    clf = clf.fit(train_X, train_y)
    importances = clf.feature_importances_
    # print(importances) 
    # print(importances.shape)
    importances_order = np.argsort(importances)
    # print(importances_order)
    # print(importances_order.shape)

    with open('data/feature_importances.pkl', 'wb') as f1:
        pkl.dump(importances, f1)
    with open('data/feature_importances_order.pkl', 'wb') as f2:
        pkl.dump(importances_order, f2)
    # print(importances[importances_order[0]], importances[importances_order[-1]])
import numpy as np
import pandas as pd


# read and return the whole data set, type is numpy.narray
def loadDataset(localPath):
    features = pd.read_csv(localPath + '/AwA2-features.txt', ' ')
    labels = pd.read_csv(localPath+'/AwA2-labels.txt', ' ')
    return np.array(features), np.array(labels)


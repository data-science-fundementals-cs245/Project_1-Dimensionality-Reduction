#-*- coding:utf-8 -*-
import argparse, random, os, sys, time


install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(install_path)  # append root dir to sys.path

from utils.data_reader import Data
from utils.solver import SVMSolver
from utils.reducer import Reducer


#################################################################################################
############################### Arguments parsing and Preparations ##############################
#################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--rtype', type=str, default='LDA', help='')

# model paras
parser.add_argument('--dims', type=int, default=100, help='target dimension')
parser.add_argument('--aim_acc', type=float, default=0.7, help='selection aimed accuracy')
parser.add_argument('--max_dim', type = float, default=100, help='selection max dimension')

opt = parser.parse_args()


####################################################################################################
####################################### Data Reader ################################################
####################################################################################################

data_set = Data()
train_X, test_X, train_y, test_y = data_set.Get_Train_Test()

####################################################################################################
##################################### Model Construction ###########################################
####################################################################################################

reducer = Reducer(opt.rtype)
train_X, test_X, train_y, test_y = reducer.reduce(train_X, test_X, train_y, test_y, opt)
solver = SVMSolver()

####################################################################################################
###################################### Training and Decoding #######################################
####################################################################################################


acc = solver.train_and_decode(train_X, test_X, train_y, test_y)
print('reducer type:\t{}\nacc:\t{}'.format(opt.rtype, acc))
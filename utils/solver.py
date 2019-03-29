#-*- coding:utf-8 -*-

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import numpy as np
import time, os, gc

class Solver():

    def __init__(self, model, loss_function, optimizer, exp_path='', logger='', device='cpu'):
        super(Solver, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device

    def train_and_validate(self, *args, **kargs):
        raise NotImplementedError

    def test(self, *args, **kargs):
        raise NotImplementedError


class SVMSolver:#类名要改

    def __init__(self):
        self.best_result = {"losses": [], "iter": 0, "v_acc": 0., "t_acc": 0., "v_loss": float('inf')}

    def train_and_decode(self, train_X, test_X, train_y, test_y):
        print('Begin to train\nX dimention:\t{}'.format(train_X.shape[1]))
        svc = SVC(kernel='rbf', class_weight='balanced', )
        c_range = np.logspace(-2, 10, 4, base=2)
        gamma_range = np.logspace(-5, 3, 5, base=2)
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=4)
        clf = grid.fit(train_X, train_y)
        acc = grid.score(test_X, test_y)
        return acc





#-*- coding:utf-8 -*-
import os, sys
root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

import sklearn.discriminant_analysis as sk_discriminant_analysis
import sklearn.decomposition as sk_decomposition
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import pickle as pkl

class Reducer:

    def __init__(self, reducer_type):
        self.type = reducer_type

    def PCA(self, train_X, test_X, dims):
        pca = sk_decomposition.PCA(n_components=dims, whiten=False, svd_solver='auto')
        pca.fit(train_X)
        print('PCA:')
        print('降维后的各主成分的方差值占总方差值得比例', pca.explained_variance_ratio_)
        print('降维后的各主成分的方差值', pca.explained_variance_)
        print('降维后的特征数', pca.n_components_)
        train_X = pca.transform(train_X)
        test_X = pca.transform(test_X)
        return train_X, test_X

    def LDA(self, train_X, test_X, train_y, dims):
        lda = sk_discriminant_analysis.LinearDiscriminantAnalysis(n_components=dims)
        lda.fit(train_X, train_y)
        print('LDA:')
        print('LDA的数据中心点：', lda.means_)
        print('LDA分类的正确率：', lda.score(train_X, train_y))
        train_X = lda.transform(train_X)
        test_X = lda.transform(test_X)

        return train_X, train_y, test_X

    def t_SNE(self, X, Y, seed, dims):
        print ("TSNE:")
        tsne = t_SNE_reduce(X, Y, seed, dims)
        reduced_X = tsne.fit_transform()
        # plot graph, only support 2d graphs
        # tsne.plot_2d_graph(reduced_X)

    def evaluation(self, selection, train_X, train_y):
        mask = np.array(sorted(selection))
        x = train_X[:, mask]
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_std, train_y, test_size=0.4)
        svc = SVC(kernel='linear', class_weight='balanced', )
        c_range = np.logspace(-2, 10, 4, base=2)
        gamma_range = np.logspace(-5, 3, 5, base=2)
        param_grid = [{'kernel': ['linear'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        clf = grid.fit(x_train, y_train)
        score = grid.score(x_test, y_test)
        return score

    def ForwardSelection(self, train_X, train_y, test_X, aim_acc, max_dims, method, step):
        n_samples = train_X.shape[0]
        choices = np.arange(n_samples)
        np.random.shuffle(choices)
        train_X = train_X[choices][:int(n_samples / 10)]
        train_y = train_y[choices][:int(n_samples / 10)]
        train_y.reshape((-1, 1))

        selection = []
        dims = 0
        if method == 'random':
            choices = np.arange(train_X.shape[1])
            np.random.shuffle(choices)
        elif method == 'importances':
            f = open('data/feature_importances_order.pkl', 'rb')
            choices = pkl.load(f)
            choices = choices[::-1]
        else:
            raise ValueError

        print("Start Forward Selection!\naim_acc={0}, max_dims={1}, method={2}, step={3}".format(aim_acc, max_dims, method, step))
        while True:
            for i in range(step):
                selection.append(choices[dims])
                dims += 1
            acc = self.evaluation(selection, train_X, train_y)
            print('dims =', dims, '; acc =', acc)
            if acc >= aim_acc or dims >= max_dims:
                break

        mask = np.array(sorted(selection))
        test_X = test_X[:, mask]
        return train_X[:, mask], train_y, test_X

    def BackwardSelection(self, train_X, train_y, test_X, aim_acc, max_dims, method, step):
        n_samples = train_X.shape[0]
        choices = np.arange(n_samples)
        np.random.shuffle(choices)
        train_X = train_X[choices][:int(n_samples / 10)]
        train_y = train_y[choices][:int(n_samples / 10)]
        train_y.reshape((-1, 1))

        selection = np.arange(train_X.shape[1]).tolist()
        d = 0
        if method == 'random':
            choices = np.arange(train_X.shape[1])
            np.random.shuffle(choices)
        elif method == 'importances':
            f = open('data/feature_importances_order.pkl', 'rb')
            choices = pkl.load(f)
        else:
            raise ValueError
        print("Start Backward Selection!\naim_acc={0}, max_dims={1}, method={2}, step={3}".format(aim_acc, max_dims, method, step))
        while True:
            for i in range(step):
                selection.remove(choices[d])
                d += 1
            acc = self.evaluation(selection, train_X, train_y)
            print('dims =', len(selection), '; acc =', acc)
            if acc <= aim_acc or len(selection) <= max_dims:
                break

        mask = np.array(sorted(selection))
        test_X = test_X[:, mask]
        return train_X[:, mask], train_y, test_X

    def reduce(self, train_X, test_X, train_y, test_y, opt):
        if self.type == 'PCA':
            dims = opt.dims
            train_X, test_X = self.PCA(train_X, test_X, dims)
            dims = opt.dims
            train_X, train_y, test_X = self.LDA(train_X, test_X, train_y, dims)
        elif self.type == 'LDA':
            dims = opt.dims
            train_X, train_y, test_X = self.LDA(train_X, test_X, train_y, dims)
        elif self.type == 'FS':
            aim_acc = opt.aim_acc
            max_dims = opt.max_dims
            method = opt.method
            step = opt.step
            train_X, train_y, test_X = self.ForwardSelection(train_X, train_y, test_X, aim_acc, max_dims, method, step)
        else:
            raise ValueError

        return train_X, test_X, train_y, test_y


class t_SNE_reduce:
    def __init__(self, features=None, labels=None, seed=0, n_dimensions=2):
        self.features = features
        self.labels = labels
        self.seed = seed
        self.n_dimensions = n_dimensions

    def fit_transform(self):
        tsne = manifold.TSNE(n_components=self.n_dimensions, init='pca', random_state=self.seed)
        X_tsne = tsne.fit_transform(self.features)
        print ("原数据维度为：{}; 降维后数据维度为：{}".format(X.shape[-1], X_tsne.shape[-1]))

        return X_tsne

    def plot_2d_graph(self, X_tsne):
        # x.min(0) -> Take the min value of each column
        # The following steps normalize each dimension
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(self.labels[i]), color=plt.cm.Set1(self.labels[i]),
                    fontdict={'weight': 'bold', 'size':9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

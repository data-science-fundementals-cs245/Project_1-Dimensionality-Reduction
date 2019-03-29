#-*- coding:utf-8 -*-
import sklearn.discriminant_analysis as sk_discriminant_analysis
import sklearn.decomposition as sk_decomposition
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

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

    def evaluation(self, selection, train_X, train_y):
        mask = np.array(sorted(selection))
        x = train_X[:, mask]
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_std, train_y, test_size=0.4)
        svc = SVC(kernel='rbf', class_weight='balanced', )
        c_range = np.logspace(-2, 10, 4, base=2)
        gamma_range = np.logspace(-5, 3, 5, base=2)
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        clf = grid.fit(x_train, y_train)
        score = grid.score(x_test, y_test)
        return score

    def ForwardSelection(self, train_X, train_y, test_X, aim_acc, max_dims):
        n_samples = train_X.shape[0]
        choices = np.arange(n_samples)
        np.random.shuffle(choices)
        train_X = train_X[choices][:int(n_samples / 10)]
        train_y = train_y[choices][:int(n_samples / 10)]
        train_y.reshape((-1, 1))

        selection = []
        dims = 0
        choices = np.arange(train_X.shape[1])
        np.random.shuffle(choices)
        while True:
            selection.append(choices[dims])
            dims += 1
            acc = self.evaluation(selection, train_X, train_y)
            print('dims =', dims, '; acc =', acc)
            if acc >= aim_acc or dims >= max_dims:
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
            train_X, train_y, test_X = self.ForwardSelection(train_X, train_y, test_X, aim_acc, max_dims)
        else:
            raise ValueError

        return train_X, test_X, train_y, test_y


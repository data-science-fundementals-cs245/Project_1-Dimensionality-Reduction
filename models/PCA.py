# coding:utf-8
import sklearn.decomposition as sk_decomposition
import numpy as np

class PCA:
    def __init__(self, X_train, y_train, X_test, y_test, flag = True):
        # flag True refers to process on the whole dataset
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.flag = flag

        if self.flag:
            self.X = np.vstack((X_train, X_test))
        else:
            self.X = X_train
    
    def transform(self, dims = 'mle'):
        pca = sk_decomposition.PCA(n_components=dims, whiten=False, svd_solver='auto')
        pca.fit(self.X)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        print('PCA:')
        print ('降维后的各主成分的方差值占总方差值得比例', pca.explained_variance_ratio_)
        print ('降维后的各主成分的方差值', pca.explained_variance_)
        print ('降维后的特征数', pca.n_components_)
        return self.X_train, self.y_train, self.X_test, self.y_test
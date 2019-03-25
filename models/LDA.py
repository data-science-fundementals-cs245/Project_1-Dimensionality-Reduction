import sklearn.discriminant_analysis as sk_discriminant_analysis
import numpy as np

class LDA:
    def __init__(self, X_train, y_train, X_test, y_test, flag = True):
        # flag True refers to process on the whole dataset
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.flag = flag

        if self.flag:
            self.X = np.vstack((X_train, X_test))
            self.y = np.vstack((y_train, y_test))
        else:
            self.X = X_train
            self.y = y_train
    
    def transform(self, dims = 100):
        lda = sk_discriminant_analysis.LinearDiscriminantAnalysis(n_components=dims)
        lda.fit(self.X, self.y)
        self.X_train = lda.transform(self.X_train) 
        self.X_test = lda.transform(self.X_test)
        print('LDA:')
        print('LDA的数据中心点：',lda.means_)
        print('LDA分类的正确率：', lda.score(self.X, self.y))   
        return self.X_train, self.y_train, self.X_test, self.y_test
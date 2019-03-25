import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

class FeatureSelection:
    def __init__(self, X_train, y_train, X_test, y_test, flag = True):
        # flag True refers to process on the whole dataset
        self.S = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.flag = flag

        # split for reduced dataset for evaluation
        if self.flag:
            self.X = np.vstack((X_train, X_test))
            self.y = np.vstack((y_train, y_test))
        else:
            self.X = X_train
            self.y = y_train 
        n_samples = self.X.shape[0]
        choices = np.arange(n_samples)
        np.random.shuffle(choices)
        self.X = self.X[choices][:int(n_samples / 10)]
        self.y = self.y[choices][:int(n_samples / 10)]
        self.y.reshape((-1, 1))


    def evaluation(self):
        mask = np.array(sorted(self.S))
        x = self.X[:, mask]
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)  # Normalization
    
        # split reduced dataset into 
        x_train, x_test, y_train, y_test = train_test_split(x_std, self.y, test_size = 0.4)

        # SVM accuracy evaluation
        # rbf核函数，设置数据权重
        svc = SVC(kernel='rbf', class_weight='balanced',)
        c_range = np.logspace(-2, 10, 4, base=2)
        gamma_range = np.logspace(-5, 3, 5, base=2)
        # 网格搜索交叉验证的参数范围，cv=3,3折交叉
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        # 训练模型
        clf = grid.fit(x_train, y_train)
        # 计算测试集精度
        score = grid.score(x_test, y_test)
        return score

    # forward selection
    def forward(self, aim_acc, max_dims):
        self.S = []
        dims = 0
        choices = np.arange(self.X.shape[1])
        np.random.shuffle(choices)
        while True:
            self.S.append(choices[dims])
            dims += 1
            acc = self.evaluation()
            print('dims =', dims, '; acc =', acc)
            if acc >= aim_acc or dims >= max_dims:
                break

        mask = np.array(sorted(self.S))
        return self.X_train[:, mask], self.y_train, self.X_test[:, mask], self.y_test

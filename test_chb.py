from utils.preprocess import loadDataset
import models.PCA
import models.LDA
import models.selection
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = loadDataset('data/ResNet101')

    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    print(X_train.shape, y_train.shape)

    # feature selection example: init the class object with X_train, y_train, X_test, y_test and flag(True refers to process on the whole dataset)
    # call function forward(aim_acc, max_dims) and returns the dataset after dimensions reduction

    # selection_model = models.selection.FeatureSelection(X_train, y_train, X_test, y_test, flag = True)
    # X_train2, y_train2, X_test2, y_test2 = selection_model.forward(0.3, 20)
    # print(X_train2.shape, y_train2.shape)


    # PCA example: (LDA is similar to this)
    # pca = models.PCA.PCA(X_train, y_train, X_test, y_test, flag = True)
    # X_train2, y_train2, X_test2, y_test2 = pca.transform(50)
    # print(X_train2.shape, y_train2.shape)

    lda = models.LDA.LDA(X_train, y_train, X_test, y_test, flag = True)
    X_train2, y_train2, X_test2, y_test2 = lda.transform(50)
    print(X_train2.shape, y_train2.shape)
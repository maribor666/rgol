import numpy as np
import pandas
import cython

from preprocess import prepareX, prepareY

class Regression(object):

    def __init__(self):
        self.theta = []
        self.nums = 0

    def _hypothesis(self, X):
        spliced = X.ravel()
        x = np.insert(spliced, 0, 1)
        matr_mult = ((-1) * self.theta).T * x
        return 1 / (1 + np.exp(matr_mult.sum()))
        
    def _loss(self, X, y):
        # cdef double summa = 0
        summa = 0
        i = 0
        for x_i, y_i in zip(X, y):
            for x_ij, y_ij in zip(x_i, y_i):
                summa += self._cost(self._hypothesis(x_ij), y_ij)
                i += 1
                print(i)
        return summa / self.nums
            

    def _cost(self, hyp, y):
        if y == 1:
            return (-1) * np.log(hyp)
        else:
            return (-1) * np.log(1 - hyp)

    def train(self, X, y, epoches=100):
        # print(X[3][3])
        # print(y[3][3])
        self.theta = np.random.randn(10)
        self.nums = len(X) * X[0].shape[0]
        # print(self.theta)
        h = self._hypothesis(X[3][3])
        # print(h)
        loss = self._loss(X, y)
        print(loss)

    def predict(self, X):
        pass

    def load_model(self):
        pass

if __name__ == '__main__':
    dataset_path = "resources/train.csv"
    df = pandas.read_csv(dataset_path)
    X = df.iloc[:, 402:].values
    Y = df.iloc[:, 2:402].values
    
    X_train, Y_train = prepareX(X), prepareY(Y)

    regr = Regression()
    regr.train(X_train, Y_train)
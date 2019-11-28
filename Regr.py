import multiprocessing as mp
import numpy as np
import pandas
# import cython

from preprocess import prepareX, prepareY

import pyximport
pyximport.install(language_level=3)
from fastFuncs import *

# mb calc new thetas after ever x_ij, y_ij
# !!!! add 1 to left of raveled X[i][i] !!!! if needed

class Regression(object):

    def __init__(self):
        self.theta = []
        self.nums = 0

    def train(self, X, y, epoches=100):
        self.cpu_count = mp.cpu_count()
        self.theta = np.random.randn(9)
        self.nums = len(X) * X[0].shape[0]
        print(f"cpu count {self.cpu_count}")
        print(len(X))
        print(X[0].shape)
        splitedX = np.array_split(X, self.cpu_count)
        splitedY = np.array_split(y, self.cpu_count)
        print("shape of splited part",splitedY[0].shape)
        # print(sum(len(el) for el in splited))
        self.pool = mp.Pool(self.cpu_count)
        input_data = [(x_part, y_part, self.theta) for x_part, y_part in zip(splitedX, splitedY)]
        res = self.pool.starmap(loss, input_data)
        summed_res = sum(res) / self.nums
        print(summed_res)
        # print(self.theta)
        # self.theta = update_theta(X, y, self.theta, self.nums)
        print(self.theta)



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
    print("end")


    # def _hypothesis(self, X):
    #     spliced = X.ravel()
    #     x = np.insert(spliced, 0, 1)
    #     matr_mult = ((-1) * self.theta).T * x
    #     return 1 / (1 + np.exp(matr_mult.sum()))
        
    # def _loss(self, X, y):
    #     # cdef double summa = 0
    #     summa = 0
    #     i = 0
    #     for x_i, y_i in zip(X, y):
    #         for x_ij, y_ij in zip(x_i, y_i):
    #             summa += self._cost(self._hypothesis(x_ij), y_ij)
    #             i += 1
    #             print(i)
    #             break
    #         break
        # return summa / self.nums
            

    # def _cost(self, hyp, y):
    #     if y == 1:
    #         return (-1) * np.log(hyp)
    #     else:
    #         return (-1) * np.log(1 - hyp)

    # def train(self, X, y, epoches=100):
        # print(X[3][3])
        # print(y[3][3])
        # self.theta = np.random.randn(10)
        # self.nums = len(X) * X[0].shape[0]
        # print(self.theta)
        # h = self._hypothesis(X[3][3])
        # print(h)
        # loss = self._loss(X, y)/
        # print(loss)
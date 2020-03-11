import pickle
import numpy as np
import pandas
import preprocess

dataset_path = "resources/train.csv"


def main():
    # it will be in main
    df = pandas.read_csv(dataset_path)
    X = df.iloc[:, 402:].values
    Y = df.iloc[:, 2:402].values
    X_train = preprocess.prepareX(X)
    Y_train = preprocess.prepareY(Y)
    print(X_train.shape)
    print(Y_train.shape)




if __name__ == '__main__':
    main()
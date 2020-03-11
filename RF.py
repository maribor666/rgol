import pickle
import numpy as np
import pandas
import preprocess
import dt_my

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

    samples_num = np.random.choice(Y_train.shape[0], Y_train.shape[0] * 2 // 3, replace=True)
    x_subset = X_train[samples_num, :]
    y_subset = Y_train[samples_num]
    print(samples_num[:20])

    tree = dt_my.build_tree(x_subset, y_subset)


    # t = np.arange(5, 10)
    # print(t)
    # samples_num = np.random.choice(t.shape[0], 3)
    # print(samples_num)
    # print(t[samples_num])

    # A[np.random.choice(A.shape[0], 2, replace=False), :]



if __name__ == '__main__':
    main()
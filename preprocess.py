import numpy as np
import pandas

dataset_path = "resources/train.csv"


def main():
    # it will be in main
    df = pandas.read_csv(dataset_path)
    X = df.iloc[:, 402:].values
    Y = df.iloc[:, 2:402].values
    # end of main code
    prepareX(X)
    # Y_train = Y.reshape(-1, 1).ravel()

    # print(Y_train.shape)
#     print(Y.shape)
    pass


def prepareY(Y):
    return Y.reshape(-1, 1).ravel()


def prepareX(X, wind_size=3):
    X = X.reshape((-1, 20, 20))
    X_train = []
    length = len(X)
    for i in range(length):
        game = X[i]
        game = np.c_[game, np.full((game.shape[0]), -1)]
        game = np.c_[np.full((game.shape[0]), -1), game]
        game = np.r_[game, [np.full((game.shape[1]), -1)]]
        game = np.r_[[np.full((game.shape[1]), -1)], game]
        windows = rolling_window(game)
        windows = windows.reshape((400, wind_size, wind_size))
        X_train.append(windows)
    x_reshaped = np.vstack((X_train))
    x_reshaped = x_reshaped.reshape((x_reshaped.shape[0], 9))
    return x_reshaped


def rolling_window(a, shape=(3, 3)):
    # rolling window for 2D array
    #   https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


if __name__ == '__main__':
    main()

import argparse
import pandas
import numpy as np

dataset_path = "resources/train.csv"
# https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray

# 1. arg parse
# 2. preprocess data (solve how exectly)
# 3. choose strategic of learning


def main():
	# parser = argparse.ArgumentParser("Some descrition")
	# parser.add_argument('dataset', default='resources/train.csv', help='train dataset')

	# args = parser.parse_args()
	# print(args.dataset)
	
	df = pandas.read_csv(dataset_path)
	X = df.iloc[:, 402:].values
	Y = df.iloc[:, 2:402].values
	X_train = X.reshape((-1, 20, 20))
	a = X_train[3]
	test = rolling_window(a, (3, 3))
	print(a)
	# print(test)
	print(test.shape)
	print(test[0][0])

	# a = np.array([[0,  1,  2,  3,  4,  5],
 #                  [6,  7,  8,  9, 10,  11],
 #                  [12, 13, 14, 15, 7,   8],
 #                  [18, 19, 20, 21, 13, 14],
 #                  [24, 25, 26, 27, 19, 20],
 #                  [30, 31, 32, 33, 34, 35]], dtype=np.int)
	# b = np.arange(36, dtype=np.float).reshape(6,6)
	# present = np.array([[7,8],[13,14],[19,20]], dtype=np.int)
	# absent  = np.array([[7,8],[42,14],[19,20]], dtype=np.int)

	# print(rolling_window(a, present.shape))
	# found = np.all(np.all(rolling_window(a, present.shape) == present, axis=2), axis=2)
	# print(np.transpose(found.nonzero()))
	# found = np.all(np.all(rolling_window(b, present.shape) == present, axis=2), axis=2)
	# print(np.transpose(found.nonzero()))
	# found = np.all(np.all(rolling_window(a, absent.shape) == absent, axis=2), axis=2)
	# print(np.transpose(found.nonzero()))
	

def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)



	


if __name__ == '__main__':
	main()

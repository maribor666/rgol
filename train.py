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
	print(test.shape)
	print(test[0][0])
	





	


if __name__ == '__main__':
	main()

import numpy as np

def main():
	mom = np.random.binomial(1, 0.5, size=(2, 2)).astype('uint8')
	dad = np.random.binomial(1, 0.5, size=(2, 2)).astype('uint8')
	mask = np.random.binomial(1, 0.5, size=(2, 2)).astype('uint8')
	child1 = mom.copy()
	print('Mom')
	print(mom)
	print('Dad')
	print(dad)
	print('Mask')
	print(mask)
	child1[mask] = dad[mask]
	print('Child')
	print(child1)


	


if __name__ == '__main__':
	main()

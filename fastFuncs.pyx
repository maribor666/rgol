cimport cython

# import numpy as np
# https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory/14657667#14657667

from libc.math cimport log, exp
cimport numpy as np
import numpy as np	

@cython.cdivision(True)
cpdef double hypothesis(theta, x): # add type hints for arguments
	# print(theta.shape)
	# print(x.shape)
	cdef double matr_mul = np.dot(theta, x)
	return 1 / (1 + exp(matr_mul))

cpdef double loss(np.ndarray X,np.ndarray Y,np.ndarray theta): # 'np' is not a cimported module. need cimport to add type hints
	# print(type(X), X.shape)
	# print(type(Y), Y.shape)
	x_test = np.ravel(X[0][0])
	y_test = Y[0][0]
	print(x_test, y_test)
	# print(type(theta))
	cdef double res = 0
	cdef double h = 0
	# res = hypothesis(theta, x_test)
	for xi, yi in zip(X, Y):
		for xij, yij in zip(xi, yi):
			h = hypothesis(theta, np.ravel(xij))
			res += cost(h, yij)
	return res

cpdef double cost(double h, y): # add type hints for arguments
	if y == 1:
		return -log(h)
	else:
		return -log(1 - h)

cpdef update_theta(np.ndarray X, Y, old_theta, nums, alfa=0.05):
	# cdef np.array[double, dim=1] new_theta = old_theta
	cdef double res = 0
	cdef double summa = 0
	cdef double[:] new_theta = np.zeros((9,))
	for theta_j in range(len(old_theta)):
		for xi, yi in zip(X, Y):
			for xij, yij in zip(xi, yi):
				xij_raveled = np.ravel(xij)
				h = hypothesis(old_theta, xij_raveled)
				res = (h - yij) * xij_raveled[theta_j]
				summa += res
		new_theta[theta_j] = old_theta[theta_j] - (alfa / nums) * summa
	return new_theta


import numpy as np

def kMeans_train(X, k, max_itertn, threshold = 0.0):

	#initiate means
	init_indices = np.random.choice(len(X), k, replace=False)

	mean = []
	for i in xrange(k):
		mean += [ X[init_indices[i]] ]

	mean = np.array(mean)

	#loop over entire datasets till max iterations are over or
								#change in all means is less than threshold for convergence
	for i in xrange(max_itertn):
		y = []
		
		distance = []
		for j in xrange(k):
			distance += [np.linalg.norm(X - mean[j], axis=1)]
		distance = np.array(distance).T

		y = np.argmin(distance, axis=1)

		new_mean = []
		for j in xrange(k):
			new_mean += [ np.mean(X[np.where(y == j)[0]], axis=0) ]
			#pos = np.array(((y-j) == 0), dtype=float)
			#new_mean += [np.mean((X.T * pos).T, axis=0)]

		new_mean = np.array(new_mean)

		if np.linalg.norm(new_mean - mean, axis=1).all() <= threshold:
			return new_mean

		else:
			mean = new_mean

	return mean


def kMeans_cluster(X, mean):

	k = len(mean)
	y = []

	for x in X:
		distance = np.zeros(k, dtype=float)

		for j in xrange(k):
			distance[j] = np.linalg.norm(x - mean[j])

		y += [np.argmin(distance)]

	return np.array(y)

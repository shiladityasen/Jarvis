from __future__ import division
import numpy as np

def sigmoid(x, derivative = False):
	if derivative:
		return x - np.square(x)
	return 1 / (1 + np.exp(-x))


def identity(x, derivative = False):
	if derivative:
		return 1.0
	else:
		return x


def error(y, t, derivative = False, form = "squared"):

	if form == "squared":
		if derivative:
			return (y - t)
		else:
			return 0.5 * np.square(y - t)

	elif form == "logistic":
		if derivative:
			return (y - t) / np.multiply(y, 1-y)
		else:
			return -(t * np.log(y) + (1 - t) * np.log(1 - y))


def forward_prop(X, W, transform = sigmoid, final_transform = identity):

	n_layers = len(W)

	X_val = [np.insert(X,0,1.0)]

	for i in xrange(n_layers - 1):
		#print i, W[i], X_val[-1]
		s = np.dot(W[i], X_val[-1])
		x = transform(s)
		x = np.insert(x,0,1.0)
		X_val += [x]

	s = np.dot(W[-1], X_val[-1])
	x = final_transform(s)
	X_val += [x]

	return X_val
	

def back_prop(X_val, W, y, transform = sigmoid, final_transform = identity, error_form = "squared"):
	
	n_layers = len(W) #2

	l = n_layers #2
	#print l
	delta = [error(X_val[n_layers], y, derivative = True, form = error_form) * final_transform(X_val[n_layers], True)]

	for l in xrange(n_layers - 1, 0, -1):
		#print l
		#print np.dot(delta[0], W[l][:,1:]), X_val[l]
		d = np.dot(delta[0], W[l][:,1:]) * transform(X_val[l][1:], True)
		delta.insert(0, d)

	return delta


def nnet_train(X, y, W, eta, l, itertn = 1, transform = sigmoid, final_transform = identity, error_form = "squared"):
	'''
		Given a training set (X, y) as numpy matrices and weight matrix configuration W, trains neural net
		using backpropagation
	'''	
	
	for i in xrange(itertn):
		for j, x in enumerate(X):
			x_val = forward_prop(x, W, transform, final_transform)
			#print x_val
			delta = back_prop(x_val, W, y[j], transform, final_transform, error_form)

			for k in range(len(W)):
				#print np.outer(delta[k], x_val[k])
				W[k] -= eta*(np.outer(delta[k], x_val[k]) + l*W[k])

	return W


def nnet_predict(X, W, transform = sigmoid, final_transform = identity):
	'''
		Given an input matrix of data points and configuration weight matrix for neural network as a 
		list of weight matrices, computes the prediction for the input data points
	'''

	X_val = forward_prop(X, W, transform, final_transform)
	return X_val[-1]


def initialize_weights(L, alpha = 1.0):

	'''
		L is a list of positive integers denoting number of neurons in each layer of neural net.
		Given alpha, function intializes a list for weight matrices for connections between layers.
		Weights are drawn from a Gaussian distribution with mean 0 and standard deviation alpha
	'''

	W = []

	for i in xrange(len(L) - 1):
		w = np.random.normal(0.0, alpha, (L[i+1], L[i]+1))
		W += [w]

	return W
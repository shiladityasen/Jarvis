from __future__ import division
import numpy as np

class nnet:

	def __init__(self, L, alpha = 1.0, eta, lambda2, transform = sigmoid, final_transform = identity, error_form ="squared"):
		self.L = L
		self.initialize_weights(self, L, alpha)
		self.eta = eta
		self.lambda2 = lambda2
		self.transform = transform
		self.final_transform = final_transform
		self.error_form = error_form


	def initialize_weights(self, alpha):

		'''
			L is a list of positive integers denoting number of neurons in each layer of neural net.
			Given alpha, function intializes a list for weight matrices for connections between layers.
			Weights are drawn from a Gaussian distribution with mean 0 and standard deviation alpha
		'''

		self.W = []

		for i in xrange(len(self.L) - 1):
			w = np.random.normal(0.0, alpha, (self.L[i+1], self.L[i]+1))
			self.W += [w]


	def nnet_train(self, X, y, itertn = 1):
		'''
			Given a training set (X, y) as numpy matrices and weight matrix configuration W, trains neural net
			using backpropagation
		'''	
		for i in xrange(itertn):
			for j, x in enumerate(X):
				x_val = forward_prop(x)
				delta = back_prop(x_val, y[j])

				for k in range(len(self.W)):
					self.W[k] -= self.eta*(np.outer(delta[k], x_val[k]) + self.lambda2 * self.W[k])

		self.W = W


	def forward_prop(X):

		n_layers = len(self.W)

		X_val = [np.insert(X,0,1.0)]

		for i in xrange(n_layers - 1):
			s = np.dot(self.W[i], X_val[-1])
			x = self.transform(s)
			x = np.insert(x,0,1.0)
			X_val += [x]

		s = np.dot(self.W[-1], X_val[-1])
		x = self.final_transform(s)
		X_val += [x]

		return X_val
		

	def back_prop(X_val, y):

		W = self.W
		transform = self.transform
		final_transform = self.final_transform
		error_form = self.error_form
		
		n_layers = len(self.W)

		l = n_layers
		delta = [error(X_val[n_layers], y, derivative = True) * self.final_transform(X_val[n_layers], True)]

		for l in xrange(n_layers - 1, 0, -1):
			#print l
			#print np.dot(delta[0], W[l][:,1:]), X_val[l]
			d = np.dot(delta[0], W[l][:,1:]) * transform(X_val[l][1:], True)
			delta.insert(0, d)

		return delta


	def nnet_predict(X, W, transform = sigmoid, final_transform = identity):
		'''
			Given an input matrix of data points and configuration weight matrix for neural network as a 
			list of weight matrices, computes the prediction for the input data points
		'''

		X_val = forward_prop(X, W, transform, final_transform)
		return X_val[-1]
		

	def sigmoid(x, derivative = False):
		if derivative:
			return x - np.square(x)
		return 1 / (1 + np.exp(-x))


	def identity(x, derivative = False):
		if derivative:
			return 1.0
		else:
			return x


	def error(y, t, derivative = False):

		if self.error_form == "squared":
			if derivative:
				return y-t
			else:
				return 0.5 * np.square(y - t)

		elif self_form == "logistic":
			if derivative:
				return (y - t) / np.multiply(y, 1-y)
			else:
				return -(t * np.log(y) + (1 - t) * np.log(1 - y))
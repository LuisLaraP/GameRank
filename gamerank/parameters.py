"""Experiments to determine regression parameters."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

import gamerank.database as db


def learningRate():
	"""Plot error vs iterations."""
	lrList = [0.001, 0.003, 0.006, 0.01, 0.03]
	epochs = 2
	xTrain = db.load('train', 'data')[:, 1:].astype(int)
	yTrain = np.squeeze(db.load('train', 'y')[:, 1:])
	for lr in lrList:
		errorCurve = np.zeros(xTrain.shape[0] * epochs)
		model = SGDRegressor(penalty='none', learning_rate='constant', eta0=lr,
			max_iter=1)
		for e in range(epochs):
			for i in range(xTrain.shape[0]):
				x = np.expand_dims(xTrain[i, :], 0)
				y = np.array([yTrain[i]])
				model.partial_fit(x, y)
				pred = model.predict(xTrain)
				errorCurve[e*xTrain.shape[0] + i] = mean_squared_error(yTrain, pred)
		plt.semilogy(range(1, errorCurve.size + 1), errorCurve, label=str(lr))
	plt.legend()
	plt.show()


def regularization():
	"""Plot error vs iterations."""
	alphaList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
	xTrain = db.load('train', 'data')[:, 1:].astype(int)
	yTrain = np.squeeze(db.load('train', 'y')[:, 1:])
	xValid = db.load('valid', 'data')[:, 1:].astype(int)
	yValid = np.squeeze(db.load('valid', 'y')[:, 1:])
	eTrain = np.zeros(len(alphaList))
	eValid = np.zeros(len(alphaList))
	for i in range(len(alphaList)):
		for _ in range(5):
			model = SGDRegressor(penalty='l2', learning_rate='constant', eta0=0.006,
				max_iter=100)
			model.fit(xTrain, yTrain)
			eTrain[i] += np.sqrt(mean_squared_error(yTrain, model.predict(xTrain)))
			eValid[i] += np.sqrt(mean_squared_error(yValid, model.predict(xValid)))
		eTrain[i] /= 5
		eValid[i] /= 5
	plt.semilogx(alphaList, eTrain, label='Training')
	plt.semilogx(alphaList, eValid, label='Validation')
	plt.legend()
	plt.show()

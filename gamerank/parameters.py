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

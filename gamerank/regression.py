"""Perform linear regression on various sets."""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

import gamerank.database as db


def dataReg():
	"""Perform regression on game metadata."""
	xTrain = db.load('train', 'data')[:, 1:].astype(int)
	yTrain = np.squeeze(db.load('train', 'y')[:, 1:])
	xValid = db.load('valid', 'data')[:, 1:].astype(int)
	yValid = np.squeeze(db.load('valid', 'y')[:, 1:])
	model = SGDRegressor(penalty='none', max_iter=100)
	model.fit(xTrain, yTrain)
	pred = model.predict(xValid)
	error = np.sqrt(mean_squared_error(yValid, pred))
	print(error)


def textReg():
	"""Preform regression on text data."""
	xTrain = db.load('train', 'text')[:, 1:].astype(int)
	yTrain = np.squeeze(db.load('train', 'y')[:, 1:])
	xValid = db.load('valid', 'text')[:, 1:].astype(int)
	yValid = np.squeeze(db.load('valid', 'y')[:, 1:])
	model = SGDRegressor(penalty='none', max_iter=100)
	model.fit(xTrain, yTrain)
	pred = model.predict(xValid)
	error = np.sqrt(mean_squared_error(yValid, pred))
	print(error)

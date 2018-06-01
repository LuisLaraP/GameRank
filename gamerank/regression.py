"""Perform linear regression on various sets."""

import sys

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


def imgReg():
	"""Perform regression on image data."""
	xTrain = db.load('train', 'img').astype(int)
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'img').astype(int)
	yValid = db.load('valid', 'y')
	iTrain = np.isin(yTrain[:, 0], xTrain[:, 0])
	xTrain = xTrain[:, 1:]
	yTrain = yTrain[iTrain, 1]
	iValid = np.isin(yValid[:, 0], xValid[:, 0])
	xValid = xValid[:, 1:]
	yValid = yValid[iValid, 1]
	model = SGDRegressor(penalty='none', max_iter=100)
	model.fit(xTrain, yTrain)
	pred = model.predict(xValid)
	error = np.sqrt(mean_squared_error(yValid, pred))
	print(error)


def textReg():
	"""Perform regression on text data."""
	xTrain = db.load('train', 'text').astype(int)
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'text').astype(int)
	yValid = db.load('valid', 'y')
	iTrain = np.isin(yTrain[:, 0], xTrain[:, 0])
	xTrain = xTrain[:, 1:]
	yTrain = yTrain[iTrain, 1]
	iValid = np.isin(yValid[:, 0], xValid[:, 0])
	xValid = xValid[:, 1:]
	yValid = yValid[iValid, 1]
	model = SGDRegressor(penalty='none', max_iter=100)
	model.fit(xTrain, yTrain)
	pred = model.predict(xValid)
	error = np.sqrt(mean_squared_error(yValid, pred))
	print(error)


def main():
	"""Script entry point."""
	if len(sys.argv) < 2:
		print('Usage: regression [data|text|covers] <options>')
		exit()
	sets = globals()[sys.argv[1] + 'Dataset'](sys.argv[1:])
	xTrain, yTrain, xValid, yValid = sets
	model = SGDRegressor(max_iter=100)
	model.fit(xTrain, yTrain)
	pTrain = model.predict(xTrain)
	eTrain = np.sqrt(mean_squared_error(yTrain, pTrain))
	pValid = model.predict(xValid)
	eValid = np.sqrt(mean_squared_error(yValid, pValid))
	print('Train error: {}\tValid error: {}'.format(eTrain, eValid))


def dataDataset(args):
	"""Return the dataset of game metadata."""
	xTrain = db.load('train', 'data')[:, 1:]
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'data')[:, 1:]
	yValid = db.load('valid', 'y')
	return xTrain, yTrain[:, 1], xValid, yValid[:, 1]


if __name__ == '__main__':
	main()

"""Perform linear regression on various sets."""

import sys

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

import gamerank.database as db


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
	print('Train error: {:.2f}\tValid error: {:.2f}'.format(eTrain, eValid))


def dataDataset(args):
	"""Return the dataset of game metadata."""
	xTrain = db.load('train', 'data')[:, 1:]
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'data')[:, 1:]
	yValid = db.load('valid', 'y')
	if len(args) == 1:
		return xTrain, yTrain[:, 1], xValid, yValid[:, 1]
	cols = np.zeros(xTrain.shape[1], dtype=bool)
	if 'esrb' in args:
		cols[0:7] = True
	if 'game_modes' in args:
		cols[7:12] = True
	if 'genres' in args:
		cols[12:32] = True
	if 'themes' in args:
		cols[32:53] = True
	xTrain = xTrain[:, cols]
	xValid = xValid[:, cols]
	return xTrain, yTrain[:, 1], xValid, yValid[:, 1]


def textDataset(args):
	"""Return the dataset of vectorized game summaries."""
	xTrain = db.load('train', 'data')
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'data')
	yValid = db.load('valid', 'y')
	iTrain = np.isin(yTrain[:, 0], xTrain[:, 0])
	xTrain = xTrain[:, 1:]
	yTrain = yTrain[iTrain, 1]
	iValid = np.isin(yValid[:, 0], xValid[:, 0])
	xValid = xValid[:, 1:]
	yValid = yValid[iValid, 1]
	if 'binary' in args:
		xTrain = np.where(xTrain > 0, 1, 0)
		xValid = np.where(xValid > 0, 1, 0)
	return xTrain, yTrain, xValid, yValid


def coversDataset(args):
	"""Return the dataset of vectorized game covers."""
	hTrain = db.load('train', 'hist')
	xTrain = db.load('train', 'img')
	yTrain = db.load('train', 'y')
	hValid = db.load('valid', 'hist')
	xValid = db.load('valid', 'img')
	yValid = db.load('valid', 'y')
	iTrain = np.isin(yTrain[:, 0], hTrain[:, 0])
	xTrain = np.hstack((hTrain[:, 1:], xTrain[iTrain, 1:]))
	yTrain = yTrain[iTrain, 1]
	iValid = np.isin(yValid[:, 0], hValid[:, 0])
	xValid = np.hstack((hValid[:, 1:], xValid[iValid, 1:]))
	yValid = yValid[iValid, 1]
	return xTrain, yTrain, xValid, yValid


if __name__ == '__main__':
	main()

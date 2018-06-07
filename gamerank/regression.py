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


def allDataset(args):
	"""Return all datasets concatenated."""
	data = dataDataset(['data', 'nostrip'])
	text = textDataset(['text', 'nostrip'])
	covers = coversDataset(['covers', 'nostrip'])
	iTrain = [x for x in data[0][:, 0]
		if x in text[0][:, 0] and x in covers[0][:, 0]]
	iValid = [x for x in data[2][:, 0]
		if x in text[2][:, 0] and x in covers[2][:, 0]]
	xTrain = np.hstack((
		data[0][np.isin(data[0][:, 0], iTrain)][:, 1:],
		text[0][np.isin(text[0][:, 0], iTrain)][:, 1:],
		covers[0][np.isin(covers[0][:, 0], iTrain)][:, 1:]
	))
	yTrain = data[1][np.isin(data[1][:, 0], iTrain), 1]
	xValid = np.hstack((
		data[2][np.isin(data[2][:, 0], iValid)][:, 1:],
		text[2][np.isin(text[2][:, 0], iValid)][:, 1:],
		covers[2][np.isin(covers[2][:, 0], iValid)][:, 1:]
	))
	yValid = data[3][np.isin(data[3][:, 0], iValid), 1]
	return (xTrain, yTrain, xValid, yValid)


def dataDataset(args):
	"""Return the dataset of game metadata."""
	xTrain = db.load('train', 'data')
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'data')
	yValid = db.load('valid', 'y')
	if len(args) == 1:
		return xTrain, yTrain[:, 1], xValid, yValid[:, 1]
	cols = np.zeros(xTrain.shape[1], dtype=bool)
	cols[0] = True
	if 'esrb' in args:
		cols[1:8] = True
	if 'game_modes' in args:
		cols[8:13] = True
	if 'genres' in args:
		cols[13:33] = True
	if 'themes' in args:
		cols[33:54] = True
	xTrain = xTrain[:, cols]
	xValid = xValid[:, cols]
	if 'nostrip' in args:
		return xTrain, yTrain, xValid, yValid
	else:
		return xTrain[:, 1:], yTrain[:, 1], xValid[:, 1:], yValid[:, 1]


def textDataset(args):
	"""Return the dataset of vectorized game summaries."""
	xTrain = db.load('train', 'data')
	yTrain = db.load('train', 'y')
	xValid = db.load('valid', 'data')
	yValid = db.load('valid', 'y')
	iTrain = np.isin(yTrain[:, 0], xTrain[:, 0])
	yTrain = yTrain[iTrain, :]
	iValid = np.isin(yValid[:, 0], xValid[:, 0])
	yValid = yValid[iValid, :]
	if 'binary' in args:
		xTrain = np.where(xTrain > 0, 1, 0)
		xValid = np.where(xValid > 0, 1, 0)
	if 'nostrip' in args:
		return xTrain, yTrain, xValid, yValid
	else:
		return xTrain[:, 1:], yTrain[:, 1], xValid[:, 1:], yValid[:, 1]


def coversDataset(args):
	"""Return the dataset of vectorized game covers."""
	hTrain = db.load('train', 'hist')
	xTrain = db.load('train', 'img')
	yTrain = db.load('train', 'y')
	hValid = db.load('valid', 'hist')
	xValid = db.load('valid', 'img')
	yValid = db.load('valid', 'y')
	if 'binary' in args:
		xTrain[:, 1:] = np.where(xTrain[:, 1:] > 0, 1, 0)
		xValid[:, 1:] = np.where(xValid[:, 1:] > 0, 1, 0)
	iTrain = [x for x in yTrain[:, 0] if x in xTrain[:, 0] and x in hTrain[:, 0]]
	xTrain = np.hstack((
		hTrain[np.isin(hTrain[:, 0], iTrain), :],
		xTrain[np.isin(xTrain[:, 0], iTrain), 1:]
	))
	yTrain = yTrain[np.isin(yTrain[:, 0], iTrain), :]
	iValid = [x for x in yValid[:, 0] if x in xValid[:, 0] and x in hValid[:, 0]]
	xValid = np.hstack((
		hValid[np.isin(hValid[:, 0], iValid), :],
		xValid[np.isin(xValid[:, 0], iValid), 1:]
	))
	yValid = yValid[np.isin(yValid[:, 0], iValid), :]
	cols = np.zeros(xTrain.shape[1], dtype=bool)
	cols[0] = True
	if 'hist' in args:
		cols[1:hTrain.shape[1]+1] = True
	if 'bow' in args:
		cols[hTrain.shape[1]+1:] = True
	if 'hist' not in args and 'bow' not in args:
		cols[:] = True
	xTrain = xTrain[:, cols]
	xValid = xValid[:, cols]
	if 'nostrip' in args:
		return xTrain, yTrain, xValid, yValid
	else:
		return xTrain[:, 1:], yTrain[:, 1], xValid[:, 1:], yValid[:, 1]


if __name__ == '__main__':
	main()

"""Rank Nintendo Switch games based on past generation games."""

import json
import sys

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

import gamerank.config as cfg
import gamerank.database as db


def main():
	"""Script entry point."""
	if len(sys.argv) == 2:
		n = int(sys.argv[1])
	else:
		n = 10
	xTrain = db.load('rank_train', 'data')
	yTrain = db.load('rank_train', 'y')
	xTest = db.load('rank_test', 'data')
	for set in ['text', 'hist', 'img']:
		nTrain = db.load('rank_train', set)
		nTest = db.load('rank_test', set)
		iTrain = [x for x in xTrain[:, 0] if x in nTrain[:, 0]]
		xTrain = np.hstack((
			xTrain[np.isin(xTrain[:, 0], iTrain), :],
			nTrain[np.isin(nTrain[:, 0], iTrain), 1:]
		))
		iTest = [x for x in xTest[:, 0] if x in nTest[:, 0]]
		xTest = np.hstack((
			xTest[np.isin(xTest[:, 0], iTest), :],
			nTest[np.isin(nTest[:, 0], iTest), 1:]
		))
	gamesPath = cfg.databasePath() + '/Games/{}.json'
	names = []
	for i in range(xTest.shape[0]):
		with open(gamesPath.format(int(xTest[i, 0])), 'r') as gameFile:
			names.append(json.load(gameFile)['name'])
	model = SGDRegressor(max_iter=100)
	model.fit(xTrain[:, 1:], yTrain[:, 1])
	pTrain = model.predict(xTrain[:, 1:])
	eTrain = np.sqrt(mean_squared_error(yTrain[:, 1], pTrain))
	print('Training error: {}'.format(eTrain))
	pTest = model.predict(xTest[:, 1:])
	sortIdx = np.argsort(pTest)
	for i in range(1, n + 1):
		idx = sortIdx[-i]
		print('{:.2f}\t{}'.format(pTest[idx], names[idx]))


if __name__ == '__main__':
	main()

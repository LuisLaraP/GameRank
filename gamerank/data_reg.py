"""Perform linear regression on game metadata."""

import numpy as np
from sklearn.linear_model import SGDRegressor

import gamerank.database as db


def main():
	"""Script entry point."""
	xTrain = db.load('train', 'data')[:, 1:].astype(int)
	yTrain = np.squeeze(db.load('train', 'y')[:, 1:])
	xValid = db.load('valid', 'data')[:, 1:].astype(int)
	yValid = np.squeeze(db.load('valid', 'y')[:, 1:])
	model = SGDRegressor(penalty='none', learning_rate='constant', max_iter=100)
	model.fit(xTrain, yTrain)
	pred = model.predict(xValid)
	error = np.sqrt(np.mean((pred - yValid) ** 2))
	print(error)


if __name__ == '__main__':
	main()
"""Functions for image preprocessing."""

import os

import cv2
import numpy as np

import gamerank.config as cfg


def extractFeatures():
	"""Extract SIFT features from all the images in the database."""
	imgPath = cfg.databasePath() + '/Covers'
	featPath = cfg.databasePath() + '/Features'
	sift = cv2.xfeatures2d.SIFT_create()
	for file in os.listdir(imgPath):
		baseName = file.split('.')[0]
		if os.path.exists(featPath + '/' + baseName + '.csv'):
			continue
		img = cv2.imread(imgPath + '/' + file, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print(file + ' skipped')
			continue
		_, desc = sift.detectAndCompute(img, None)
		if len(desc.shape) < 2:
			print(file + ' skipped')
			continue
		np.savetxt(featPath + '/' + baseName + '.csv', desc, fmt='%.2f')


def vectorizeImages():
	"""Transform game covers into a vector suitable for regression."""
	extractFeatures()

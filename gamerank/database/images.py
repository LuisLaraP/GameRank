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
		if os.path.exists(featPath + '/' + baseName + '.npy'):
			continue
		img = cv2.imread(imgPath + '/' + file, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print(file + ' skipped')
			continue
		_, desc = sift.detectAndCompute(img, None)
		if desc is None or len(desc.shape) < 2:
			print(file + ' skipped')
			continue
		desc = desc / np.sum(desc, axis=1)[:, np.newaxis]
		desc = np.clip(desc, None, 0.2)
		desc = desc / np.sum(desc, axis=1)[:, np.newaxis]
		np.save(featPath + '/' + baseName, desc, allow_pickle=False)


def vectorizeImages():
	"""Transform game covers into a vector suitable for regression."""
	extractFeatures()

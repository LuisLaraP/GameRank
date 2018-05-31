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
		img = cv2.imread(imgPath + file, cv2.IMREAD_GRAYSCALE)
		_, desc = sift.detectAndCompute(img, None)
		baseName = file.split('.')[0]
		np.savetxt(featPath + baseName + '.csv', desc)

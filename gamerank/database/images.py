"""Functions for image preprocessing."""

import json
import os

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import gamerank.config as cfg


def clusterFeatures():
	"""Perform clustering on the extracted features."""
	featPath = cfg.databasePath() + '/Features'
	with open(cfg.databasePath() + '/Sets.json', 'r') as setsFile:
		idList = json.load(setsFile)['train']
	arrays = (np.load(featPath + '/' + str(id) + '.npy', allow_pickle=False)
		for id in idList if os.path.exists(featPath + '/' + str(id) + '.npy'))
	features = np.vstack(arrays)
	print(features.shape)
	print('Loading complete')
	config = cfg.readConfig()
	k = config.getint('Images', 'n_clusters')
	model = MiniBatchKMeans(n_clusters=k, batch_size=50000, verbose=True,
		compute_labels=False)
	model.fit(features)
	np.savetxt(cfg.databasePath() + '/Centers.csv', model.cluster_centers_)


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
	print('Extracting features')
	extractFeatures()
	print('Clustering features')
	clusterFeatures()

"""Functions for image preprocessing."""

import json
import os

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import gamerank.config as cfg


def buildBagOfWords():
	"""Transform image features to a BoW vector."""
	featPath = cfg.databasePath() + '/Features'
	config = cfg.readConfig()
	k = config.getint('Images', 'n_clusters')
	centers = np.loadtxt(cfg.databasePath() + '/Centers.csv')
	searcher = NearestNeighbors(n_neighbors=1, n_jobs=-1)
	searcher.fit(centers)
	with open(cfg.databasePath() + '/Sets.json', 'r') as setsFile:
		sets = json.load(setsFile)
	curSet = 'train'
	idList = [x for x in sets[curSet]
		if os.path.exists(featPath + '/{}.npy'.format(x))]
	data = np.zeros((len(idList), k + 1), dtype=int)
	data[:, 0] = idList
	for i in range(len(idList)):
		features = np.load(featPath + '/{}.npy'.format(idList[i]))
		assignments = searcher.kneighbors(features, return_distance=False)
		for item in assignments:
			data[i, item + 1] += 1
	np.savetxt(cfg.databasePath() + '/train_img.csv', data, fmt='%d')


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
	if os.path.exists(cfg.databasePath() + '/Centers.csv'):
		print('Cluster centers found. Skipping.')
	else:
		clusterFeatures()
	print('Vectorizing')
	buildBagOfWords()

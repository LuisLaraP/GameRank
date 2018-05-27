"""Functions for data preprocessing."""

import json
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import gamerank.config as cfg


def encodeData():
	"""Encode game metadata."""
	config = cfg.readConfig()
	esrbCode = config.get('Preprocessing', 'esrb').split(',')
	modesCode = config.get('Preprocessing', 'game_modes').split(',')
	genresCode = config.get('Preprocessing', 'genres').split(',')
	themesCode = config.get('Preprocessing', 'themes').split(',')
	modesCode = [int(x) for x in modesCode]
	genresCode = [int(x) for x in genresCode]
	themesCode = [int(x) for x in themesCode]
	dim2 = sum([len(esrbCode), len(modesCode), len(genresCode), len(themesCode)])
	with open(cfg.databasePath() + '/Sets.json', 'r') as setsFile:
		sets = json.load(setsFile)
	for set in sets:
		idList = sets[set]
		data = np.zeros((len(idList), dim2), dtype=int)
		for i in range(len(idList)):
			gamePath = cfg.databasePath() + '/Games/{}.json'.format(idList[i])
			with open(gamePath, 'r') as gameFile:
				gameData = json.load(gameFile)
			esrb = gameData['esrb']['rating'] if 'esrb' in gameData else 1
			data[i, :len(esrbCode)] = \
				[1 if esrb - 1 == i else 0 for i in range(len(esrbCode))]
			p = len(esrbCode)
			modes = gameData['game_modes'] if 'game_modes' in gameData else []
			data[i, p:p+len(modesCode)] = encodeMultiLabel(modes, modesCode)
			p += len(modesCode)
			genres = gameData['genres'] if 'genres' in gameData else []
			data[i, p:p+len(genresCode)] = encodeMultiLabel(genres, genresCode)
			p += len(genresCode)
			themes = gameData['themes'] if 'themes' in gameData else []
			data[i, p:p+len(themesCode)] = encodeMultiLabel(themes, themesCode)
		np.savetxt(cfg.databasePath() + '/{}.csv'.format(set), data, fmt='%d')


def encodeMultiLabel(values, code):
	"""Encode list of values as a several-of-K vector."""
	vec = np.zeros(len(code))
	for v in values:
		vec[code.index(v)] = 1
	return vec


def vectorizeSummaries():
	"""Create word count matrix from game summaries."""
	with open(cfg.configPath() + '/Vocabulary.txt', 'r') as inFile:
		vocab = [x.rstrip() for x in inFile]
	config = cfg.readConfig()
	params = config['Text']
	vectorizer = CountVectorizer(**params)
	vectorizer.set_params(vocabulary=vocab)
	with open(cfg.databasePath() + '/Sets.json', 'r') as setsFile:
		sets = json.load(setsFile)
	ids = []
	summaries = []
	for id in sets['train']:
		with open(cfg.databasePath() + '/Games/{}.json'.format(id), 'r') as inFile:
			gameData = json.load(inFile)
		if 'summary' in gameData:
			ids.append(id)
			summaries.append(gameData['summary'])
	vectorizer.fit(summaries)
	train = np.zeros((len(ids), 1 + len(vectorizer.get_feature_names())))
	train[:, 0] = ids
	train[:, 1:] = vectorizer.transform(summaries).todense()
	np.savetxt(cfg.databasePath() + '/trainSummaries.csv', train, fmt='%d')
	ids = []
	summaries = []
	for id in sets['valid']:
		with open(cfg.databasePath() + '/Games/{}.json'.format(id), 'r') as inFile:
			gameData = json.load(inFile)
		if 'summary' in gameData:
			ids.append(id)
			summaries.append(gameData['summary'])
	valid = np.zeros((len(ids), 1 + len(vectorizer.get_feature_names())))
	valid[:, 0] = ids
	valid[:, 1:] = vectorizer.transform(summaries).todense()
	np.savetxt(cfg.databasePath() + '/validSummaries.csv', valid, fmt='%d')
	ids = []
	summaries = []
	for id in sets['test']:
		with open(cfg.databasePath() + '/Games/{}.json'.format(id), 'r') as inFile:
			gameData = json.load(inFile)
		if 'summary' in gameData:
			ids.append(id)
			summaries.append(gameData['summary'])
	test = np.zeros((len(ids), 1 + len(vectorizer.get_feature_names())))
	test[:, 0] = ids
	test[:, 1:] = vectorizer.transform(summaries).todense()
	np.savetxt(cfg.databasePath() + '/testSummaries.csv', test, fmt='%d')

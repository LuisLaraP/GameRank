"""Functions for accessing and modifying the database."""

import json
import os
import random
import requests

from igdb_api_python.igdb import igdb

import gamerank.config as cfg


def download():
	"""Download game data from IGDB."""
	if not os.path.exists(cfg.databasePath() + '/Games'):
		os.makedirs(cfg.databasePath() + '/Games')
	config = cfg.readConfig()
	fields = config['Database']['fields'].split(',')
	platforms = config['Database']['platforms']
	api = igdb(config['Database']['api_key'])
	res = api.games({
		'fields': fields,
		'filters': {
			'[release_dates.platform][any]': platforms,
			'[aggregated_rating][gte]': 0
		},
		'scroll': 1,
		'limit': 50
	})
	for game in res.body:
		filename = cfg.databasePath() + '/Games/{}.json'.format(game['id'])
		with open(filename, 'w') as outFile:
			json.dump(game, outFile, indent='\t')
	nPages = round(int(res.headers['X-Count']) / 50)
	for _ in range(nPages):
		scrolled = api.scroll(res)
		if type(scrolled.body) is list:
			for game in scrolled.body:
				filename = cfg.databasePath() + '/Games/{}.json'.format(game['id'])
				with open(filename, 'w') as outFile:
					json.dump(game, outFile, indent='\t')


def downloadCovers():
	"""Download covers for the games present in the database."""
	coverPath = cfg.databasePath() + '/Covers/'
	dataPath = cfg.databasePath() + '/Games/'
	if not os.path.exists(coverPath):
		os.makedirs(coverPath)
	for file in os.listdir(dataPath):
		with open(dataPath + file, 'r') as inFile:
			data = json.load(inFile)
		try:
			url = 'https:' + data['cover']['url'].replace('thumb', 'cover_big')
			coverFile = coverPath + url.split('/')[-1]
			if os.path.isfile(coverFile):
				print('Skipping {}'.format(data['name']))
			else:
				response = requests.get(url)
				with open(coverFile, 'wb') as outFile:
					outFile.write(response.content)
		except KeyError:
			print('Skipping {}: no cover.'.format(data['name']))


def splitDatabase():
	"""Split database into train, validation and test sets."""
	config = cfg.readConfig()
	dataPath = cfg.databasePath() + '/Games'
	idList = []
	for file in os.listdir(dataPath):
		if os.path.isfile(dataPath + '/' + file):
			idList.append(int(file.split('.')[0]))
	trainSize = int(config.getfloat('Database', 'train_size') * len(idList))
	validSize = int(config.getfloat('Database', 'valid_size') * len(idList))
	sets = {}
	sets['train'] = random.sample(idList, trainSize)
	idList = [x for x in idList if x not in sets['train']]
	sets['valid'] = random.sample(idList, validSize)
	sets['test'] = [x for x in idList if x not in sets['valid']]
	with open(cfg.databasePath() + '/Sets.json', 'w') as outFile:
		json.dump(sets, outFile)

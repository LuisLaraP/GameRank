"""Functions for database management."""

import json
import os
import random

import gamerank.config as cfg


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

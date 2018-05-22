"""Functions for accessing and modifying the database."""

import json
import os

from igdb_api_python.igdb import igdb

import gamerank.config as cfg


def download():
	"""Download all data from IGDB."""
	if not os.path.exists(cfg.databasePath()):
		os.makedirs(cfg.databasePath())
	config = cfg.readConfig()
	platforms = config['Database']['platforms']
	api = igdb(config['Database']['api_key'])
	res = api.games({
		'fields': 'id',
		'filters': {
			'[release_dates.platform][any]': platforms,
			'[aggregated_rating][gte]': 0
		},
		'scroll': 1,
		'limit': 50
	})
	nPages = round(int(res.headers['X-Count']) / 50)
	idList = [x['id'] for x in res.body]
	for _ in range(nPages):
		scrolled = api.scroll(res)
		if type(scrolled.body) is list:
			idList.extend([x['id'] for x in scrolled.body])
	with open(cfg.databasePath() + '/Ids.json', 'w') as idxFile:
		json.dump(idList, idxFile)

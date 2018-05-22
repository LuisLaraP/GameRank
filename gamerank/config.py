"""Configuration loading and saving."""

import appdirs


appName = 'GameRank'


def databasePath():
	"""Return the absolute path to the root directory of the database."""
	return appdirs.user_data_dir(appName) + '/database'

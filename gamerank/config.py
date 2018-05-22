"""Configuration loading and saving."""

import configparser

import appdirs


appName = 'GameRank'


def createConfig():
	"""Create a default configuration file."""
	path = appdirs.user_config_dir(appName) + '/GameRank.cfg'
	open(path, 'a').close()


def databasePath():
	"""Return the absolute path to the root directory of the database."""
	return appdirs.user_data_dir(appName) + '/database'


def readConfig():
	"""Read the configuration file and return the parsed contents."""
	path = appdirs.user_config_dir(appName) + '/GameRank.cfg'
	config = configparser.ConfigParser()
	config.read(path)
	return config

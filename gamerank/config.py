"""Configuration loading and saving."""

import configparser

import appdirs


appName = 'GameRank'
configFilename = appdirs.user_config_dir(appName) + '/GameRank.cfg'


def createConfig():
	"""Create a default configuration file."""
	open(configFilename, 'a').close()


def databasePath():
	"""Return the absolute path to the root directory of the database."""
	return appdirs.user_data_dir(appName) + '/database'


def readConfig():
	"""Read the configuration file and return the parsed contents."""
	config = configparser.ConfigParser()
	config.read(configFilename)
	return config

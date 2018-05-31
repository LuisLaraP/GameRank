"""Setup script."""

from setuptools import find_packages, setup


setup(
	name='GameRank',
	description='A video game ranking system powered by machine learning.',
	version='0.3.0',
	author='Luis Alejandro Lara Patiño',
	author_email='luislpatino@gmail.com',
	license='GPLv3',
	packages=find_packages(),
	python_requires='>=3',
	install_requires=[
		'appdirs>=1.4.3',
		'igdb_api_python>=1',
		'numpy',
		'scikit-learn',
		'scipy'
	],
	entry_points={
		'console_scripts': [
			'data-reg = gamerank.regression:dataReg',
			'dl-covers = gamerank.database.download:downloadCovers',
			'dl-data = gamerank.database.download:downloadData',
			'param-alpha = gamerank.parameters:regularization',
			'param-lr = gamerank.parameters:learningRate',
			'pp-data = gamerank.database.preprocessing:encodeData',
			'pp-text = gamerank.database.preprocessing:vectorizeSummaries',
			'split-db = gamerank.database.management:splitDatabase',
			'text-reg = gamerank.regression:textReg'
		]
	}
)

"""Setup script."""

from setuptools import find_packages, setup


setup(
	name='GameRank',
	description='A video game ranking system powered by machine learning.',
	version='0.0.0',
	author='Luis Alejandro Lara Patiño',
	author_email='luislpatino@gmail.com',
	license='GPLv3',
	packages=find_packages(),
	python_requires='>=3',
	install_requires=[
		'appdirs>=1.4.3',
		'igdb_api_python>=1'
	]
)

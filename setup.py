from setuptools import setup

setup(name='round',
	version='0.1',
	description='Stellar rotation with PyMC3 and Exoplanet',
	url='http://github.com/tagordon/round',
	author='Tyler Gordon',
	author_email='tagordon@uw.edu',
	licens='MIT',
	packages=['round'],
	install_requires=[
        	'numpy',
		'matplotlib',
		'astropy',
		'exoplanet',
		'pymc3',
		'theano',
		'corner'
	],
	zip_safe=False)

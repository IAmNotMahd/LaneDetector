#!/usr/bin/env python3

import sys
from setuptools import setup, find_packages

install_requires = []

setup(name='lane_detector',
	version='0.1',
	description="A python wrapper for CARMERA's SCNN implementation for traffic lane detection",
	author="Mahd Aamir Khan",
	author_email='khan-psip@carmera.co',
	url='https://github.com/carmeraco/LaneDetector',
	packages=find_packages(),
	license='MIT License',
	install_requires=install_requires
)
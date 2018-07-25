#!/usr/bin/env python3

import sys
from setuptools import setup, find_packages

install_requires = []

setup(name='lane_detector',
      version='1.2.3',
      description="A Python client for CARMERAs ML toolkit",
      author="Chase Nicholl",
      author_email='chase@carmera.com',
      url='https://github.com/carmeraco/LaneDetector',
      packages=find_packages(),
      install_requires=install_requires)
#!/usr/bin/env python3
import sys
from setuptools import setup

install_requires = ['numpy',
                    'torch',
                    'dscribe', 
                    'ase']

packages = ['aenet_gpr',
            'aenet_gpr.inout',
            'aenet_gpr.src',
            'aenet_gpr.util',]

if __name__ == '__main__':

    assert sys.version_info >= (3, 0), 'python>=3 is required'

    with open('./README.md', 'rt', encoding='UTF8') as f:
        long_description = f.read()

    with open('aenet_gpr/__init__.py', 'r') as init_file:
        for line in init_file:
            if "__version__" in line:
                version = line.split()[2].strip('\"')
                break

setup(name='aenet-gpr',
      version=version,
      description='Atomistic simulation tools based on Gaussian processes',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/atomisticnet/aenet-gpr',
      license='MPL-2.0',
      packages=packages,
      install_requires=install_requires,
      python_requires='>=3',
      keywords=['machine learning', 'potential energy surface', 'aenet', 'data augmentation'],
)

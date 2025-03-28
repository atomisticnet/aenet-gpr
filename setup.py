#!/usr/bin/env python3
import sys
from setuptools import setup, find_packages


install_requires = ['numpy',
                    'torch',
                    'dscribe',]

packages = ['aenet_GPR',
            'aenet_GPR.inout',
            'aenet_GPR.src',
            'aenet_GPR.util',]

scripts = ['scripts/aenet_GPR.py', ]

if __name__ == '__main__':

    assert sys.version_info >= (3, 0), 'python>=3 is required'

    with open('__init__.py', 'r') as init_file:
        for line in init_file:
            if "__version__" in line:
                version = line.split()[2].strip('\"')
                break

setup(name='aenet_gpr',
      description='Atomistic simulation tools based on Gaussian processes',
      url='https://github.com/atomisticnet/aenet_gpr',
      version=version,
      license='MPL-2.0',
      packages=packages,
      install_requires=install_requires,
)

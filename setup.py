import sys
from setuptools import setup, find_packages

install_requires = ['numpy<2.0',
                    'torch',
                    'dscribe', 
                    'ase']

packages = ['aenet_gpr']

if __name__ == '__main__':

    assert sys.version_info >= (3, 0), 'python>=3 is required'

    with open('aenet_gpr/__init__.py', 'r') as init_file:
        for line in init_file:
            if "__version__" in line:
                version = line.split()[2].strip('\"')
                break

setup(name='aenet-gpr',
      version=version,
      description='Atomistic simulation tools based on Gaussian processes',
      url='https://github.com/atomisticnet/aenet-gpr',
      license='MPL-2.0',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3',
      keywords=['machine learning', 'potential energy surface', 'aenet', 'data augmentation'],
)

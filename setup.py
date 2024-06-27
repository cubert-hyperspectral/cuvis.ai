from setuptools import setup, find_packages
from pathlib import Path

NAME = 'cuvis_ai'
VERSION = '0.0.1'
DESCRIPTION = 'CUVIS AI Toolset.'

REQUIREMENTS = {}

with open(Path(__file__).parent / "requirements.txt", "r") as reqfile:
    REQUIREMENTS["install"] = reqfile.readlines()

setup(
    name=NAME,
    python_requires='>= 3.9',
    version=VERSION,
    url='https://www.cubert-hyperspectral.com/',
    license='Apache License 2.0',
    author='Cubert GmbH, Ulm, Germany',
    author_email='hanson@cubert-gmbh.com',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS['install'],
    include_package_data=True,
    packages=find_packages() # Automatically find packages and subpackages)
)

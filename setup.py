from setuptools import setup

NAME = 'cuvis_ai'
VERSION = '0.0.1'

DESCRIPTION = 'CUVIS AI Toolset.'

REQUIREMENTS = {
    'install': [
        'cuvis-il == 3.2.1',
        'numpy',
        'scikit-learn',
        'matplotlib'
    ],
}

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
    packages=[NAME]
)

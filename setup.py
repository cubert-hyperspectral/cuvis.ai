from setuptools import setup, find_packages
from pathlib import Path
from pathlib import Path

NAME = 'cuvis_ai'
VERSION = '0.0.1'
DESCRIPTION = 'CUVIS AI Toolset.'

REQUIREMENTS = {}

here = Path(__file__).parent

with open(Path(__file__).parent / "requirements.txt", "r") as reqfile:
    REQUIREMENTS["install"] = reqfile.readlines()


def __createManifest__(subdirs):
    """inventory all files in path and create a manifest file"""

    relative_paths = [Path(path).relative_to(here) for path in subdirs]

    single_files = [here / 'README.md']

    rel_single_files = [path.relative_to(here)
                        for path in single_files]

    with open(here / "MANIFEST.in", "w") as manifest:
        manifest.writelines(
            "include {}  \n".format(" ".join(str(rel_single_files))))


add_il = here / "cuvis_ai"

__createManifest__([add_il])

setup(
    name=NAME,
    python_requires='>= 3.10',
    version=VERSION,
    url='https://www.cubert-hyperspectral.com/',
    license='Apache License 2.0',
    author='Cubert GmbH, Ulm, Germany',
    author_email='hanson@cubert-gmbh.com',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS['install'],
    include_package_data=True,
    packages=find_packages()
)

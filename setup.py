import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 7, 7):
    sys.exit('project requires Python >= 3.7.7')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

with open('README.rst') as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name='',
    description='sentiment_analysis',
    long_description=readme,
    author='cem',
    author_email='huseyincemayaz@gmail.com',
    packages=find_packages(exclude=['*tests*']),
    python_requires='==3.7.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
        ]
    },
)
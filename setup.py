from setuptools import setup, find_packages

VERSION = '0.0.13'
DESCRIPTION = 'Provide tools to complete the tasks of the drone load competition.'

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Setting up
setup(
    name="droneload",
    version=VERSION,
    author="Hugo Degeneve",
    author_email="<hugo.degeneve@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=requirements,
    keywords=['python', 'video', 'drone', 'load', 'competition'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
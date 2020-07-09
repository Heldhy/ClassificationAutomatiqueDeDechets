from setuptools import setup, find_packages

with open("requirements.txt", "r") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name='WasteClassifier',
    python_requires='>=3.8',
    version='1.6.0',
    packages=find_packages(exclude=['tests.*', 'tests']),
    url='',
    license='',
    author='danae',
    author_email='danae.marmai@octo.com',
    description='A package to predict the type of waste from a picture',
    install_requires=requirements
)

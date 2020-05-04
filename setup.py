from setuptools import setup

setup(
    name='WasteClassifier',
    version='1.0.0',
    packages= find_packages(exclude=['tests.*', 'tests']),
    url='',
    license='',
    author='danae',
    author_email='danae.marmai@octo.com',
    description='A package to predict the type of waste from a picture',
    install_requires=['tensorflow', 'matplotlib', 'scikit-learn', 'numpy', 'opencv-python', 'pytest']
)

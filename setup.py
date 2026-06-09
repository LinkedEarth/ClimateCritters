from setuptools import setup, find_packages

version = '0.9.0'

setup(
    name='climatecritters',
    version=version,
    author='Jordan Landers, Alex James, Julien Emile-Geay',
    author_email='lplander@usc.edu',
    package_dir={"": "."},
    packages=find_packages(),
    description='A package for generating synthetic paleoclimate data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    url='http://github.com/LinkedEarth/ClimateCritters',
    download_url='https://github.com/LinkedEarth/ClimateCritters/tarball/'+version,
)
from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='neulib',
    packages=find_packages(include=['neulib']),
    version='0.1.3',
    description='Tool for data preprocess in ML.',
    author='Roman Fitzjalen',
    author_email='romaactor@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/kndahl/neufarm',
    include_package_data = True,
    install_requires=['numpy', 'pandas', 'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)
from setuptools import find_packages, setup

setup(
    name='neulib',
    packages=find_packages(include=['neulib']),
    version='0.1.0',
    description='Tool for data preprocess in ML.',
    author='kndahl',
    license='MIT',
    install_requires=['numpy', 'pandas', 'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)
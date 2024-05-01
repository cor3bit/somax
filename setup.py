import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='somax',
    version='0.0.1.dev',
    author='Nick Korbit',
    description='SOMAX: Second-Order Methods for Machine Learning in JAX',
    long_description=read('README.md'),
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=setuptools.find_packages(exclude=[
        'artifacts',
        'benchmarks',
        'examples',
        'scripts',
        'tests',
    ]),
    platforms='any',
    # python_requires='>=3.10',
    # install_requires=[
    #     'jax',
    # ],
)

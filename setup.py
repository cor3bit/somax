import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='python-somax',
    version='0.0.1',
    author='Nick Korbit',
    description='SOMAX: Second-Order Methods for Machine Learning in JAX',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(
        where='src',
        exclude=[
            'examples',
            'tests',
        ],
    ),
    platforms='any',
    # python_requires='>=3.10',
    install_requires=[
        'jaxopt>=0.8.2',
    ],
)

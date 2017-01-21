# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='taco_salad',
    version='0.0.1',

    description='A package with approach to combine classical maschine learn '
                'algorithms in a stacked layer way.',
    long_description=long_description,

    url='https://github.com/mbrner/taco_salad',

    author='Mathis Boerner',
    author_email='mathis.boerner@tu-dortmund.de',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    # What does your project relate to?
    keywords='multivariate distribution evaluation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib>=1.4',
        'scikit-learn>=0.18.1',
        'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

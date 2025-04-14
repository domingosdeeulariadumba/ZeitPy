from setuptools import setup, find_packages

# Loading README.md file
with open('README.md') as f:
    readme = f.read()

# Loading the LICENSE file
with open('LICENSE') as f:
    _license = f.read()

setup(
    name = 'zeitpy',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'scipy>=1.13.1',
        'scikit-learn>=1.5.1',
        'matplotlib>=3.9.2',
        'seaborn>=0.13.2',
        'statsmodels>=0.14.2'        
    ],
    author = 'Domingos de Eulária Dumba',
    author_email = 'domingosdeeulariadumba@gmail.com',
    description = ('A package for time series analysis — from EDA to '\
                   'forecasting and performance assessment.'),
    long_description = readme,
    license = _license,
    url = 'https://github.com/domingosdeeulariadumba.com/ZeitPy',
    classifiers = [
        'License :: OSI Aproved :: MIT',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Data Scientists/Analysts',
        'Intended Audience :: Researchers',
        'Intended Audience :: Developers',        
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires = '>=3.12.7'
)
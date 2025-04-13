from setuptools import setup, find_packages

# Loading README.md file
with open('README.md') as f:
    readme = f.read()

# Loading the LICENSE file
with open('LICENSE') as f:
    license = f.read()

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
    author='Domingos de Eulária Dumba',
    author_email='domingosdeeulariadumba@gmail.com',
    description = ('A package for time series analysis — from EDA to '\
                   'Forecasting and Performance Assessment.'),
    long_description=readme,
    license=license,
    url = 'https://github.com/domingosdeeulariadumba.com/ZeitPy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Data Scientists/Analysts',
        'Intended Audience :: Researchers',
        'Intended Audience :: Developers',        
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires = '>=3.12.7'
)
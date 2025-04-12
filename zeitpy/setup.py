from setuptools import setup, find_packages

setup(
    name = 'zeitpy',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'statsmodels>=0.12.0'        
    ],
    author='Domingos de Eulária Dumba',
    author_email='domingosdeeulariadumba@gmail.com',
    description = ('A package for time series analysis — from EDA to '\
                   'Forecasting and Performance Assessment'),
    url = 'https://github.com/domingosdeeulariadumba.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent' 
        'Intended Audience :: Data Scientists/Analysts',
        'Intended Audience :: Researchers',
        'Intended Audience :: Developers',        
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires = '>=3.12.7'
)
from setuptools import setup, find_packages


setup(
    name='lambdamart',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.2.4',
        'scikit-learn>=0.24.2',
        'pandas>=1.3.0',
        'scipy>=1.15.2',
        'pytest>=8.2.0'
    ]
)

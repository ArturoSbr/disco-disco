# Import functions
from setuptools import setup, find_packages

# Declare description of package
desc = """disco-disco is a Python package for causal inference using sharp
regression discontinuity designs.
"""

# Run setup
setup(
    name='disco-disco',
    packages=find_packages(
        include=['disco-disco']
    ),
    version='0.1.0',
    description=desc,
    author='Arturo Soberón',
    license='MIT',
    install_requires=[
        'numpy>=1.23.2',
        'pandas>=1.4.4',
        'scikit-learn>=1.1.2'
    ]
)
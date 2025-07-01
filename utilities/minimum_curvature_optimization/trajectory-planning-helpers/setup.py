import setuptools

"""
Package on pypi.org can be updated with the following commands:
python3 setup.py sdist bdist_wheel
sudo python3 -m twine upload dist/*
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='trajectory-planning-helpers',
    version='0.76',
    url='https://github.com/TUMFTM/trajectory_planning_helpers',
    author="Alexander Heilmeier, Tim Stahl, Fabian Christ",
    author_email="alexander.heilmeier@tum.de, stahl@ftm.mw.tum.de",
    description="Useful functions used for path and trajectory planning at TUM/FTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.18.1',
        'scipy>=1.3.3',
        'quadprog~=0.1.11',
        'matplotlib>=3.3.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ])

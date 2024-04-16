from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use BERT for tracking problems in HEP analysis"

setup(
    name="TrackingBert",
    version="1.0.0",
    description="Library for using BERT in HEP analysis",
    long_description=description,
    author="Xiangyang Ju, Andris Huang",
    license="MIT License",
    keywords=["BERT", "LLM", "HEP", "analysis", "machine learning"],
    url="https://github.com/xju2/TrackingBert",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[

    ],
    package_data = {
        "TrackingBert": ["*.py"]
    },
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    scripts=[],
)

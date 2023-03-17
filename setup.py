#!/usr/bin/env python
# coding=utf-8
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sequence_model_train",
    version="0.2.90",
    author="ZhangLe",
    author_email="zhanglenlp@gmail.com",
    description="time series model for training sequence dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://doc.com",
    project_urls={
        "Bug Tracker": "https://doc.com",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages("."),
    install_requires=[
      'torch>=1.6.0',
      'pandas>=1.3.5',
      'numpy>=1.21.5',
      'DateTime>=4.4',
      'xgboost>=1.2.0',
      'scikit-learn>=1.0.2',
      'keras-tcn>=3.4.0',
      'ngboost>=0.3.13',
      'statsmodels>=0.13.5',
      'matplotlib>=3.1.1',
      'pickleshare>=0.7.5',
      'cloudpickle>=1.2.2',
      'optuna>=2.10.0'],
    python_requires=">=3.6",
)

# dune-classification
Pattern recognition algorithms for the classification of reconstructed particles for the DUNE experiment.

The project can be summarised as having 2 aims:

1) Developing algorithms to distinguish between track-like and shower-like particles by mining meaningful features and using a naive-Bayes/XGBoost approach
2) Classifying indicidual particles within events and their interaction types

A package is provisioned to help with writing scripts to acheive these aims.

## Installing the package
As the project is ongoing and the code is probably only meaningful to a select few, the package has not been submitted to PyPI.

**Installing in editable mode within a venv is recommended**.
### To install
using conda:
1) create venv using: `conda create --name myenv`
2) activate venv:     `conda activate myenv`
3) clone repo
4) install using pip: `pip install -e .`

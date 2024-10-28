# PROMT

## Contents

## Acknowledgements

This work was forked from [paste](https://github.com/raphael-group/paste) repository.

## Data Availability

Datasets are collected from [cellxgene](https://cellxgene.cziscience.com/collections/31937775-0602-4e52-a799-b6acdd2bac2e).

All preprocessed datasets are available on
[zenodo](https://zenodo.org/records/13997882).

## Reproducibility

### Environment setup

The code is tested on ubuntu 20.04 with python `3.10`. It is recommended to use a python virtual environment to run the code.

### Library Dependencies

- scipy `1.11.3`
- anndata `0.10.9`
- matplotlib `3.9.2`
- numpy `1.26.4`
- pandas `2.2.3`
- scanpy `1.10.3`
- scikit-learn `1.5.2`
- torch `2.5.0`
- stalign

  ```bash
  pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
  ```

### How to reproduce the results

Tutorial is available at [PROMT_tutorial.ipynb](./PROMT_tutorial.ipynb) in this repository.

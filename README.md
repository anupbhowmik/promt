# PROMT

## Contents

- [Acknowledgements](#acknowledgements)
- [Data Availability](#data-availability)
- [Use PROMT](#use-promt)
  - [Environment setup](#environment-setup)
  - [Library Dependencies](#library-dependencies)
- [How to reproduce the results](#how-to-reproduce-the-results)

## Acknowledgements

Some of the utility functions are taken from [raphael-group/paste](https://github.com/raphael-group/paste) repository.

## Data Availability

Datasets are collected from [cellxgene](https://cellxgene.cziscience.com/collections/31937775-0602-4e52-a799-b6acdd2bac2e).

All preprocessed datasets are available on
[zenodo](https://zenodo.org/records/13997882).

## Use PROMT

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
- mygene `3.2.2`
- anndata `0.10.9`
- stalign

  ```bash
  pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
  ```

- spatialde `1.1.3`

  ```bash
  pip install spatialde
  ```

To utilize the GPU, install [PyTorch](https://pytorch.org/) with CUDA support.

#### Requirements for paste

`paste` takes quite a long time to run especially for such a large dataset.

-

#### Requirements for stalign

-

## How to reproduce the results

> Note: This part is not finalized yet. It will be updated as the files are organized and will be finalized in the latest release.

- Create a directory `data/Mouse_brain_MERFISH` and place the preprocessed data files in it.

- Tutorials are available in python notebooks.

### Generate Results for Table 1 - 3 and Figure 2

Run the notebook `alignment_performance_metric.ipynb` with the following data pairs:

| Source                        | Target                        |
| ----------------------------- | ----------------------------- |
| adata4wk_donor_id_1_slice_0   | adata4wk_donor_id_4_slice_1   |
| adata4wk_donor_id_1_slice_0   | adata4wk_donor_id_4_slice_2   |
| adata4wk_donor_id_4_slice_1   | adata4wk_donor_id_4_slice_2   |
| adata24wk_donor_id_10_slice_0 | adata24wk_donor_id_10_slice_1 |
| adata24wk_donor_id_10_slice_0 | adata24wk_donor_id_10_slice_2 |
| adata24wk_donor_id_10_slice_1 | adata24wk_donor_id_10_slice_2 |
| adata90wk_donor_id_2_slice_0  | adata90wk_donor_id_5_slice_1  |
| adata90wk_donor_id_2_slice_0  | adata90wk_donor_id_5_slice_2  |
| adata90wk_donor_id_5_slice_1  | adata90wk_donor_id_5_slice_2  |

###

# PROMT

## Data Availability
https://cellxgene.cziscience.com/collections/31937775-0602-4e52-a799-b6acdd2bac2e

## PROMT Reproducibility
The PROMT pipeline tutorial is available at [PROMT_tutorial.ipynb](./PROMT_tutorial.ipynb) in this repository.


Make sure to split the `Anndata` object into different slices
```python
import scanpy as sc
adata_full = sc.read_h5ad('path/to/adata.h5ad')
adata_donor_MsBrainAgingSpatialDonor_1 = adata_full[adata_full.obs.donor_id == 'MsBrainAgingSpatialDonor_1']
```

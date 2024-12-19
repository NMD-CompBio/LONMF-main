# LONMF_py
LONMF: A non-negative matrix factorization model based on graph Laplacian and optimal transmission for Paired single-cell multi-omics data integration

This is the Python implementation of the LONMF algorithm. Note that this implementation supports GPU acceleration.

## 1. Installation
You can use the following command to install LONMF:
```
pip install LONMF
```

## 2. Usage
LONMF accepts a muon object as input and populates the obsm and uns fields with embeddings and dictionaries, respectively.
The pre-processed 10X Multiome demo dataset is available for download here. It is recommended that a GPU be used to run the model.
To initialize and run the LONMF model, the following code should be used:

```
from LONMF.model import LONMF
import mudata as md
import scanpy as sc

# Load data into a Muon object.
mdata = md.read_h5mu("my_data.h5mu")

# Initialize and train the model.
model = LONMF(latent_dim=15)
model.train(mdata)

# Visualize the embedding with UMAP.
sc.pp.neighbors(mdata, use_rep="W_OT")
sc.tl.umap(mdata)
sc.pl.umap(mdata)
```

'LONMF' class also has other methods, you can use the 'help' or '?' command for more details explanations of the methods.


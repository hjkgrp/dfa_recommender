DFA recommender
==============================
[//]: # (Badges)
[![CI](https://github.com/chenruduan/dfa_recommender/actions/workflows/CI.yaml/badge.svg)](https://github.com/chenruduan/dfa_recommender/actions/workflows/CI.yaml)
[![Documentation Status](https://readthedocs.org/projects/dfa-recommender-hjkgrp/badge/?version=latest)](https://dfa-recommender-hjkgrp.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/chenruduan/dfa_recommender/branch/main/graphs/badge.svg?token=A1A3S0E2F9)](https://codecov.io/gh/chenruduan/dfa_recommender/branch/main)


### System-specific density functional recommender
The idea is to recommend a density functional approximation (DFA) in the realm of density functional theory (DFT) that best approximate the properties that would be obtained by the reference method (coupled cluster, experiment, etc.). Here we assume we have the 3D geometry optimized at B3LYP to start with, since it has been observed that the optimized geometry obtained by DFT and mor accurate methods (e.g., CASPT2) is [very similar](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d1cp04885f). Therefore, we use a [density fitting approach](https://www.nature.com/articles/s41467-020-20471-y) to decompose the electron density as node features on a molecular graph.
![Recommender approach](https://github.com/chenruduan/dfa_recommender/blob/main/DFARec.png)

Due to the ambiguity of the definition of the best DFA (multiple DFAs perform similarly well in practice), we frame this question as a "regress-then-classify" task. We build transfer learning models to directly predict the absolute difference of the result of the reference and a DFA. We do this for all candidate DFAs in the pool, where 48 DFAs that span multiple rungs of "Jacob's ladder" are considered in this workflow by default. Finally we sort the predicted differences and select the DFA that yields the lowest predicted difference.
![Recommender workflow](https://github.com/chenruduan/dfa_recommender/blob/main/DFARecWorkflow.png)

### Installation
1. Clone this repo `git clone https://github.com/hjkgrp/dfa_recommender.git`
2. Setup a conda environment with the provided yaml file `conda env create -f dfa_recommender/devtools/conda-envs/test_env.yaml`
3. Pip installation `conda activate dfa_rec && cd dfa_recommender && pip install -e .`
4. Test everything works as expected `python setup.py test`

The installation should take less than 10 minutes on a normal laptop.

### File structure
```
./dfa_recommender/
├── __init__.py
├── __pycache__
├── _version.py
├── data
├── dataset.py
├── df_class.py
├── df_utils.py
├── evaluate.py
├── ml_utils.py
├── net.py
├── predict.py
├── sampler.py
├── scripts
├── tests
├── tutorials-submitted
└── vat.py
```
* All `.py` files are Python functions, where the comments and use case are available at the `API` section in the [`readthedoc` document](https://dfa-recommender-hjkgrp.readthedocs.io/en/latest/).
* `data` contains csv file, featurization, trained models, and the optimized geometries for *`VSS-452`* and *`CSD-76`* set.
* `scripts` contains Python scripts for quick model training and electron density processing.
* `tests` contains unit testing of the DFA recommender.
* `tutorials-submitted` contains Jupyter notebooks that reproduce all the results in the paper. Please refer to at the `Tutorial` section in the [`readthedoc` document](https://dfa-recommender-hjkgrp.readthedocs.io/en/latest/) for the details.

### Dependency
* Pytorch (1.10.0), Psi4 (1.6.1), Pandas (1.4.3), Scikit-learn (1.1.1)
* Tested on MacOS (M1, M1pro), Linux (RedHat 7)

### Citation
```
@Article {dfa_recommender,
author = {Duan, Chenru and Nandy, Aditya and Meyer, Ralf and Arunachalam, Naveen and Kulik, Heather J.},
title = {A Transferable Recommender Approach for Selecting the Best Density Functional Approximations in Chemical Discovery},
journal = {arXiv},
url = {https://arxiv.org/abs/2207.10747},
doi = {https://doi.org/10.48550/arXiv.2207.10747},
year = {2022},
}
```

### Reproduction instructions
All the results reported in the paper above should be reproduced by the Jupyter notebooks at `dfa_recommender/tutorials-submitted`. These notebooks also have code blocks demonstrating the usage of our models.

### Developers

Chenru Duan at HJK Group@MIT


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

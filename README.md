dfa_recommender
==============================
[//]: # (Badges)
[![CI](https://github.com/chenruduan/dfa_recommender/actions/workflows/CI.yaml/badge.svg)](https://github.com/chenruduan/dfa_recommender/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/chenruduan/dfa_recommender/branch/main/graphs/badge.svg?token=A1A3S0E2F9)](https://codecov.io/gh/chenruduan/dfa_recommender/branch/main)


### System-specific density functional recommender.
The idea is to recommend a density functional approximation (DFA) in th realm of density functional theory (DFT) that best approximate the properties that would be obtained by the reference method (coupled cluster, experiement, etc.). Here we assume we have the 3D geometry optimized at B3LYP to start with, since it has been observed that the optimized geometry obtained by DFT and mor accurate methods (e.g., CASPT2) is [very similar](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d1cp04885f). Therefore, we can use a [density fitting approach](https://www.nature.com/articles/s41467-020-20471-y) to decompose the electron density as node features on a molecular graph.
![Recommender approach](https://github.com/chenruduan/dfa_recommender/blob/main/DFARec.png)
Due to the ambiguity of the definition of the best DFA (multiple DFAs perform similarly well in practice), we frame this question as a "regress-then-classify" task.
![Recommender workflow](https://github.com/chenruduan/dfa_recommender/blob/main/DFARecWorkflow.png)

### Copyright

Copyright (c) 2022, Chenru Duan


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

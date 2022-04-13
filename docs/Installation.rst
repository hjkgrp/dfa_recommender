Installation
============

We recommend installation via source under a conda environment. 


Installing from Source
----------------------

To install from source, you can do the following::

    git clone https://github.com/chenruduan/dfa_recommender.git
    cd dfa_recommender
    conda create -f devtools/conda-envs/test_env.yaml
    conda activate dfa_rec
    pip install -e .

Please remember to activate the conda environment (dfa_rec) everytime when you want to use dfa_recommender.

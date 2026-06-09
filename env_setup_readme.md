# Setting up the environment

To set up the environment, run the following commands in the terminal:

```
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f env.yml

conda activate cc_env
conda install ipykernel    
python -m ipykernel install --user --name=cc_env
```

This will create a conda environment named `cc_env` with all the necessary packages installed. The environment can be activated by running `conda activate cc_env`.

Note: the commands about `ipykernel` are necessary to make the environment available in Jupyter notebooks.
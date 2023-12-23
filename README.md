# diffusions-for-sbi
Diffusion Generative Modeling and Posterior Sampling in Simulation-Based Inference

## Install requirements

- Create a conda environment with python version 3.10.
- Install packages via `pip install -e .` inside the repository folder.

## Contents

Score models, loss functions and diffused priors:
- `nse.py`: original conditional neural score estimator and corresponding (DSM) loss function.
- `pf_npse.py`: neural posterior score estimator for variable context set sizes.
- `vp_diffused_priors.py`: analytic diffused (uniform) prior for a VP-SDE diffusion process.

Utils:
- `plot_utils.py`: plot the 2D posteriors from samples
- `sm_utils.py`: train function for `NSE` or `PF-NPSE` models

SBI tasks:
- Toy Example: 2D Gaussian Simulator and Uniform prior - infer the mean.

Notebooks:
- `npse_intro.ipynb`: how learn and sample from the posterior of the toy example.
- `DGM4SBI_intro.ipynb`: how learn and sample from the posterior of the toy example in the context of "single" or "tall" context data (with variable sizes).

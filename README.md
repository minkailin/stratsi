# **Stratified and vertically-shearing streaming instabilities**

## Summary

## Paper
Lin (2021)

## Requirements
* [`DEDALUS`](https://dedalus-project.org/)
* [`EIGENTOOLS`](https://github.com/DedalusProject/eigentools)

## Core solvers
* `stratsi_1fluid.py`  
Complete code for solving the one-fluid  linearized equations.  
* `stratsi_params.py`  
Input parameters and vertical profiles for the two-fluid problem.
* `stratsi_eqm.py`  
For solving the two-fluid equilibrium horizontal velocity profiles.
* `stratsi_pert.py`  
For solving the two-fluid linearized equations.

## Utilities
* `run_problem.sh`  
For running the complete two-fluid problem (computing equilibrium then solving the linearized equations).
* `eigenproblem.py`  
Copied from the EIGENTOOLS package.
* `stratsi_maxvshear.py`  
For computing the largest vertical shear rate in the disk and its location.
* `stratsi_plot_eqm.py`  
For checking the two-fluid equilibrium horizontal velocity profiles by comparing the right-hand-side and left-hand-sides of the equilibrium equations.

## Generic plotting
*  `stratsi_plot.py`
Main plotting tool. Compares one- and two-fluid results.

## Special plotting
* `stratsi_plot_visc.py`  
Compares viscous results to inviscid results. Require inviscid results under folder "novisc".
* compare_etas/`stratsi_compare_eta.py`  
Compare results from two different eta values.
* compare_stokes/`stratsi_compare_stokes.py`  
Compare results from two different stokes numbers.
* compare_Z/`stratsi_compare_Z.py`  
Compare results from two different metallicities. 

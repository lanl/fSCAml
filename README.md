# fSCA ML: fractional snow covered area (fSCA) gradient boosted machine learning (ML) \
## EIDR O#:4961

This repository contains Jupyter notebooks and python scripts to train and predict fractional snow covered area in the Upper Colorado River Basin. We used a gradient boosted trees approach and high-resolution snow process modeling from SnowModel (Liston and Elder 2006) results to predict fSCA at 1000m resolution for the entire watershed. We matched the domain area for NoahMP model results at the same 1000m resolution. 

This repository contains the following:

1. A jupyter notebook for training the fSCA ML and producing a trained model object.
1. A trained and pickled model object (fSCA_model.pkl)
1. A jupyter notebook for predicting fSCA using the model object.
1. A python script for data chunking. 
1. A jupyter notebook for pre-processing the input data.

For more information or questions, please contact:
- Ryan Crumley, rcrumley@lanl.gov
- Katrina Bennett, kbennett@lanl.gov
- Cade Trotter, ctrotter@lanl.gov

**This research was funded by the NASA Terrestrial Hydrology Program**

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

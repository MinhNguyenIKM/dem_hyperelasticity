--------------------------------------------------------------------
Paper: 
A deep energy method for finite deformation hyperelasticity

Authors: Vien Minh Nguyen-Thanh, Xiaoying Zhuang, Timon Rabczuk

European Journal of Mechanics - A/Solids
Available online 25 October 2019, 103874
https://doi.org/10.1016/j.euromechsol.2019.103874
--------------------------------------------------------------------
Setup:

1. Setup environment with conda: conda create -n demhyper python=3.7

2. Switch to demhyper environment to start working with dem: source activate demhyper

3. Install numpy, scipy, matplotlib: pip install numpy scipy matplotlib

4. Install pytorch and its dependencies: conda install pytorch-cpu torchvision-cpu -c pytorch

5. Install pyevtk for view in Paraview: pip install pyevtk

Optional:

To use fem to compare the results with fem, we recommend to install fenics
Commands:
1. conda config --add channels conda-forge

2. !conda install fenics

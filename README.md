# FNN
Source code of ["Efficient and quantum-adaptive machine learning with fermion neural networks"](https://arxiv.org/abs/2211.05793), [Phys. Rev. Applied 20, 044002 (2023)](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.20.044002).

For the classical examples, run train.sh to load dataset and train the FNN, run Inter_Vis.py to show the results (Fig. 2, 6, 8). The original data of these figures are shown in 'results/MNIST/Fig2'.

For the non-interacting topological insulator, run Data_Ins_d.py (Data_Ins.py) to generate the dataset of disordered models (clean models), run train.sh to train the FNN, run Inter_Vis.py to show the results (Fig. 5(a)) and plot the phase diagrams (Fig. 3(a)(c)), run Inter_ModChe.py to plot the spectrums (Fig. 3(b)(d)), run Inter_LogFlow.py to show the logic flow (Fig. 12), run Inter_ConBandAna.py for the generative example (Fig. 5(b)), run QLTTrain.py to generate the QLT dataset and train the ANN for comparison (Fig. 5(a)). 

For the strongly correlated systems, run FK_Data_QPT.py (FK_Data.py) to generate the dataset of quantum phase transition (metal-insulator transition), run fk_train.sh to train the FNN, run FK_Inter_Vis.py to show the results and plot the phase diagrams (Fig. 4, 11), run FK_DMFT.py (FK_DMFT_kspace.py) to compute the order parameters of Falicov-Kimbal model at 2D by DMFT in real (momentum) space and plot the phase diagrams (Fig. 9, 10).

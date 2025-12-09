# CV-GHZ\_photon-sub



\## Summary

This project provides the Python code to replicate the plots provided in the journal article:

Z. M. Léger, A. Brodutch, and A. Helmy, "Entanglement enhancement in multimode integrated circuits," Phys. Rev. A 97, 062303 (2018). DOI: 10.1103/PhysRevA.97.062303, arXiv:1803.05981



\## Authorship

This code was written and is currently being upkept by Zacharie Leger. The code was originally reviewed by Aharon Brodutch under a research project supervised by Amr Helmy. 



\## Packages

Necessary packages: numpy, matplotlib, seaborn (for ideal plotting colors), qutip



QuTiP package documentation can be found here: https://qutip.org/



J. R. Johansson, P. D. Nation, F. Nori: “QuTiP 2: A Python framework for the dynamics of open quantum systems.”, Comp. Phys. Comm. 184, 1234 (2013). 

J. R. Johansson, P. D. Nation, F. Nori: “QuTiP: An open-source Python framework for the dynamics of open quantum systems.”, Comp. Phys. Comm. 183, 1760–1772 (2012).



\## Usage 

\# MainFunctions.py

Provide the necessary for the generation of the quantum state, the continuous variable (CV Greenberger–Horne–Zeilinger (GHZ) state, the split single mode squeezed vacuum state used to calculate the logarithmic-negativity of the relevant GHZ state, the transformed sigle photon subtraction operator, and calculations for the logarithmic negativity for pure and mixed states.



\# plot.py

Provides the plot functions.



\# Squeezing Parameter Results.py

This calculates and plots the logarithmic negativity as a function of the squeezing parameter r.



\# k Parameter Results.py

This calculates and plots the gain in logarithmic negativity as a function of the ratio of the squeezing parameters k.



\# Mode Results

This calculates and plots the gain in logarithmic negativity as a function of the for the number of modes, N.



Alert about notation: N in the context of the paper refers to the number of mode, while in the python script the variable N refers to the number of photons considered in the Fock space of each mode.



\# Loss Results.py

This calculates and plots the gain in logarithmic negativity as a function of the symmetric loss applied to all four modes of a CV GHZ state.




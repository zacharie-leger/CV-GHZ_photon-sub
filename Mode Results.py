# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 17:18:25 2025

@author: zacharie-leger
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from MainFunctions import *
from plot import *

#*** Define Working Parameters ***
N3 = 5         # Dimentions of the Hilbert space in the Fock state basis for splittings needing 3 modes
N4 = 4         # Dimentions of the Hilbert space in the Fock state basis for splittings needing 4 modes

"""
To define the squeezing parameters r_1 and r_2 we instead use r and k where
we define the krr as the k parameter in the equations r_1=r/(k+1) and r_2=kr/(k+1)

In this code r represents the r equation above and krr represent the parameter 
k in the equation above. The use defining the parameters r_1 and r_2 in terms of
k and r because altering the k parameter does not change the the logarithmic
negativety pre photon subtraction (see DOI: 10.1103/PhysRevA.97.062303 for details)
"""
r = 0.2


krr = [0.,0.24,0.82] 
tot=len(krr)

modes = np.arange(4,41,4)#np.array([4,8,16])
M = len(modes)
split3 = [modes*3/4,modes/2,modes/4]
split4C = [modes/2,modes/4,modes/4,modes/4,modes/4]
split4D = [modes/4,modes/4,modes/2,modes/2,modes/4]

# Define the solution space for the logarithmic negativity (we consider 4 bipartite splitings)
GHZN0ln = np.zeros([8,tot,2*M])
GHZN1ln = np.zeros([8,tot,2*M])


"""
*** Calculate the logarithmic negativity ***
We are calculating the logarithmic negativity for the state phi0 and phi1 

Here psi0 is a single mode squeezed vacuum state (SMSV) that is symmetrically splif via
an array of beamsplitter and phi1 is the photon subtraccted version of the same state.
For this calculation

    N3 and N4   is the number of photons conted in the Fock space basis of each mode
    modes       is the number of modes the single mode squeezed vacuum state 
                is split into in a symmetric beamsplitter operation
    r           is the squeezing parameter for the SMSV operator
    split3 and 
    split4C     is the number of mode collected in second the bipartite splitting 
                used evaluate the logarithmic negativity
    splitD      is the number of mode traced out in the bipartite splitting used 
                evaluate the logarithmic negativity
                
    The function SS represents the annhilation operator after applying the transformation 
    S_A^dagger (k r/(k+1)) a_A  S_A (k r/(k+1)) = a_A cosh(k r/(k+1)) -  a_A^dagger sinh(k r/(k+1))
"""

for k in range(3):
    for i in range(tot):
        for j in range(M):
            if k==1:
                phi0 = Nms_SMSV3(N3,modes[j]-2,-r,split3[k][j]-1)
                phi1 = (SS(N3,krr[i]*r/(krr[i]+1),3)*phi0).unit() 
     
                GHZN0ln[k][i][2*j] = logneg(N3, phi0, 0)
                GHZN1ln[k][i][2*j] = logneg(N3, phi1, 0)
                
                phi0 = Nms_SMSV3(N3,modes[j],-r,split3[k][j])
                phi1 = (SS(N3,krr[i]*r/(krr[i]+1),3)*phi0).unit() 
     
                GHZN0ln[k][i][2*j+1] = logneg(N3, phi0, 0)
                GHZN1ln[k][i][2*j+1] = logneg(N3, phi1, 0)
            else:
                phi0 = Nms_SMSV3(N3,modes[j],-r,split3[k][j])
                phi1 = (SS(N3,krr[i]*r/(krr[i]+1),3)*phi0).unit() 
     
                GHZN0ln[k][i][j] = logneg(N3, phi0, 0)
                GHZN1ln[k][i][j] = logneg(N3, phi1, 0)

        print('Done'+str(i+k*tot+1)+'/'+str(8*tot))

for k in range(5):
    if k==0 or k==1 or k==2:
        for i in range(tot):
            for j in range(M):
                phi0 = Nms_SMSV4(N4,modes[j],-r,split4C[k][j],split4D[k][j])
                phi1 = (SS(N4,krr[i]*r/(krr[i]+1),4)*phi0).unit()
                
                GHZN0ln[k+3][i][j] = logneg(N4, phi0, 1)
                GHZN1ln[k+3][i][j] = logneg(N4, phi1, 1)
                
            print('Done'+str(i+(k+3)*tot+1)+'/'+str(8*tot))
            
    if k==3 or k==4:
        for i in range(tot):
            for j in range(M):
                phi0 = Nms_SMSV4(N4,modes[j],-r,split4C[k][j],split4D[k][j])
                phi1 = (SS(N4,krr[i]*r/(krr[i]+1),4)*phi0).unit()
                
                GHZN0ln[k+3][i][j] = logneg(N4, phi0, 2)
                GHZN1ln[k+3][i][j] = logneg(N4, phi1, 2)
                
            print('Done'+str(i+(k+3)*tot+1)+'/'+str(8*tot))

Gain = (GHZN1ln-GHZN0ln)/GHZN0ln



#----------------------------***Plot the Results***----------------------------

Gain_vs_N(r, Gain, modes, krr)
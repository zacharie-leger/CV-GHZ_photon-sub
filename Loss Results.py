# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:36:26 2025

@author: zachl
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from MainFunctions import *
from plot import *

N = 5 # number of photon allowed in a mode
r = np.array([0.2]) #squeezing parameter

nn = len(r)

krr=np.array([0.82])

loss_param = np.linspace(np.pi/2,np.pi,51)
loss = np.cos(loss_param)**2
loss_dim = len(loss)

# Define the solution space for the logarithmic negativity (we consider 4 bipartite splitings)
GHZN0ln = np.zeros([8,nn,loss_dim])
GHZN1ln = np.zeros([8,nn,loss_dim])
for p in range(nn):  
    #create the 4 mode GHZ state
    psi = GHZ4(N,r[p], krr[0]) 
    
    #Turn into density matrix
    rho = psi*psi.dag() 
    
    #*** Calculate the logarithmic negativity ***
    for n in range(loss_dim):
        rho0 = rho
        for m in range(4):
            rho0 = tensor(rho0, basis(N,0)*basis(N,0).dag()) #tensor in an additional mode to the state
            Utr = U_TR(N,loss_param[n],510+10*m) #create the BS unitary between mode m and the vacuum mode
            rho0 = Utr*rho0*Utr.dag() #apply the unitary on the density matrix
            rho0 = rho0.ptrace([0,1,2,3]) #trace out the mode with the loss
        #print(rho0)
        rho1 = (ten(410,N)*rho0*ten(411,N)).unit() # create the photon subtracted version of the state
        
        #calculate the log. neg.
        for k in range(8):
            GHZN0ln[k][p][n] = logneg_mixed(N, rho0, k)
            GHZN1ln[k][p][n] = logneg_mixed(N, rho1, k)
            print('Done'+str(n*8+k+1)+'/'+str(8*loss_dim))

Gain_vs_loss(loss, GHZN0ln, GHZN1ln, r, krr)



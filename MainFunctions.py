# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:49:50 2025

@author: zacharie-leger
"""

from qutip import *
import numpy as np
import numpy.linalg as la


"""
Function calculating the logarithmic negativity.
    N       the number of photons in the Hilbert space
    Psi     the pure CV GHZ state being analized in the Fock state basis 
    i       denotes the splitting we are chosing to calculate
"""
def logneg(N,Psi,i): 
    rho = Psi*Psi.dag()
    #AB-C Splitting
    if i==0:
        rho = partial_transpose(rho, [1,1,0]) #, method="sparse"
        rho2 = np.zeros([N**3,N**3])
        for i in range(N**3):
            rho2[:][i] = rho[:][i] #Convert to numpy array 
            
    #Tr(D) AB-C
    elif i==1:
        rho = rho.ptrace([0,1,2])
        rho = partial_transpose(rho, [0,0,1])
        rho2 = np.zeros([N**3,N**3])
        for i in range(N**3):
            rho2[:][i] = rho[:][i] #Convert to numpy array

    #Tr(AB) C-D
    elif i==2:
        rho = rho.ptrace([2,3])
        rho = partial_transpose(rho, [0,1])
        rho2 = np.zeros([N**2,N**2])
        for i in range(N**2):
            rho2[:][i] = rho[:][i] #Convert to numpy array
    
    elif i==3:
        rho = rho.ptrace([0,1])
        rho = rho.unit()
        rho = partial_transpose(rho, [0,1])
        rho2 = np.zeros([N**2,N**2])
        for i in range(N**2):
            rho2[:][i] = rho[:][i] #Convert to numpy array
    #find Eignvelues
    #lda = rho.eigenenergies()
    lda = la.eigvals(rho2) 
    
    #Log.-neg. is the log base 2 of the sum of the eigenvalues 
    LN = np.log2(np.sum(np.abs(lda)))
    return LN

#Log-Neg function defined for mixed states
def logneg_mixed(N,Rho,i): 
    #AB_0-C_3 Splitting
    if i==0:
        Rho = partial_transpose(Rho, [1,0,0,0]) #, method="sparse"
        Rho2 = np.zeros([N**4,N**4])
        for i in range(N**4):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
    
    #AB_1-C_2 Splitting
    elif i==1:
        Rho = partial_transpose(Rho, [1,1,0,0]) #, method="sparse"
        Rho2 = np.zeros([N**4,N**4])
        for i in range(N**4):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
            
    #AB_2-C_1 splitting
    elif i==2:
        Rho = partial_transpose(Rho, [1,1,1,0])
        Rho2 = np.zeros([N**4,N**4])
        for i in range(N**4):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array   
    
    #Tr(D_1) AB_0-C_2 Splitting
    elif i==3:
        Rho = Rho.ptrace([0,1,2]).unit()
        Rho = partial_transpose(Rho, [1,0,0])
        Rho2 = np.zeros([N**3,N**3])
        for i in range(N**3):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
    
    #Tr(D_1) AB_1-C_1
    elif i==4:
        Rho = Rho.ptrace([0,2,3])
        Rho = partial_transpose(Rho, [1,1,0])
        Rho2 = np.zeros([N**3,N**3])
        for i in range(N**3):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
        
    #Tr(D_2) AB_0-C_1 Splitting
    elif i==5:
        Rho = Rho.ptrace([0,1]).unit()
        Rho = partial_transpose(Rho, [1,0])
        Rho2 = np.zeros([N**2,N**2])
        for i in range(N**2):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
    
    #Tr(AB_0) C_1-D_2 Splitting
    elif i==6:
        Rho = Rho.ptrace([1,2,3]).unit()
        Rho = partial_transpose(Rho, [1,0,0])
        Rho2 = np.zeros([N**3,N**3])
        for i in range(N**3):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
                   
    #Tr(AB_1) C_1-D_1
    elif i==7:
        Rho = Rho.ptrace([2,3])
        Rho = partial_transpose(Rho, [1,0])
        Rho2 = np.zeros([N**2,N**2])
        for i in range(N**2):
            Rho2[:][i] = Rho[:][i] #Convert to numpy array
        
    #find Eignvelues
    lda = la.eigvals(Rho2) 
    
    #Log.-neg. is the log base 2 of the sum of the eigenvalues 
    LN1 = np.log2(np.sum(np.abs(lda)))
    return LN1

"""
Notes:
From here on out N will refer to the dimention of the Hilbert space that we are considering and
r will refer to the squeezing parameter.

Here we define the Nms_SMSV3 and Nms_SMSV4 as the single mode squeesed vacuum state approximated where:
    N       is the number of photons conted in the Fock space basis of each mode
    modes   is the number of modes this single mode squeezed vacuum state is split into in a symmetric beamsplitter 
            operation
    r       is the real non-negative squeezing parameter of the single mode squeezed state
    C       is the number of mode collected in second the bipartite splitting used evaluate the logarithmic negativity
    D       is the number of mode traced out in the bipartite splitting used evaluate the logarithmic negativity
"""



def Nms_SMSV3(N,modes,r,C):
    if C<=modes:
        sq = (tensor(destroy(N),qeye(N),qeye(N)) + np.sqrt(modes-C-1)*tensor(qeye(N),destroy(N),qeye(N)) + np.sqrt(C)*tensor(qeye(N),qeye(N),destroy(N)))/np.sqrt(modes)
    else:
        print("The grouping of modes is larger then the number of modes themselves")
    Psi = squeezing(sq, sq, r)*tensor(basis(N,0),basis(N,0),basis(N,0))
    return Psi.unit()

def Nms_SMSV4(N,modes,r,C,D):
    if C+D<=modes:
        sq = (tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + np.sqrt(modes-C-D-1)*tensor(qeye(N),destroy(N),qeye(N),qeye(N)) + np.sqrt(C)*tensor(qeye(N),qeye(N),destroy(N),qeye(N))+ np.sqrt(D)*tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(modes)
    else:
        print("The grouping of modes is larger then the number of modes themselves")
    Psi = squeezing(sq, sq, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
    return Psi.unit()



"""
This is the single photon subtraction operator after applying a squeezing operator 
        S_A^dagger (r) a_A  S_A (r) = a_A cosh(r) -  a_A^dagger sinh(r)

    N       is the number of photons conted in the Fock space basis of each mode

    r       is the real non-negative squeezing parameter of the single mode squeezed state
    split   refers to the number modes we need to consider to calculate the logarithmic negativity. In our case we 
            will have 3 modes if no modes are being traced out and 4 modes in we are tracing out any number of modes
"""
def SS(N, r, split):
    if split==4:
        an = np.cosh(r)*tensor(destroy(N), qeye(N), qeye(N), qeye(N)) #4 modes
        cr = np.sinh(r)*tensor(create(N), qeye(N), qeye(N), qeye(N))  #4 modes
    elif split==3:
        an = np.cosh(r)*tensor(destroy(N), qeye(N), qeye(N)) #3 modes
        cr = np.sinh(r)*tensor(create(N), qeye(N), qeye(N)) #3 modes
    return an-cr

"""
To applyloss to the state we define the four mode CV GHZ state explicitly
instead of the squeezing parameters r_1 and r_2 we use r and k where
r_1=r/(k+1) and r_2=kr/(k+1)

In this code 
    r       represents the r equation above 
    krr     represent the parameter k in the equation above 
    
The use defining the parameters r_1 and r_2 in terms of k and r because altering
the k parameter does not change the the logarithmic negativety pre photon 
subtraction (see DOI: 10.1103/PhysRevA.97.062303 for details)
"""

#Defining the 4 mode GHZ state 
def GHZ4(N,r,krr):
    sq_a1 = (ten(410,N) + ten(420,N) + ten(430,N) +  ten(440,N))/2.
    sq_a2 = (3*ten(410,N) - ten(420,N) - ten(430,N) - ten(440,N))/np.sqrt(12.)
    sq_a3 = (2*ten(420,N) - ten(430,N) - ten(440,N))/np.sqrt(6.)
    sq_a4 = (ten(430,N) -  ten(440,N))/np.sqrt(2.)
    Psi = squeezing(sq_a1, sq_a1, -r/(1+krr))*squeezing(sq_a2, sq_a2, krr*r/(1+krr))*squeezing(sq_a3, sq_a3, krr*r/(1+krr))*\
        squeezing(sq_a4, sq_a4, krr*r/(1+krr))*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0)) 
    return Psi.unit()

#Define the beam splitter unitary for loss
def U_TR(N,theta,i):
    U_Tr = squeezing(-ten(i,N),ten(551, N),2*theta-np.pi)
    return U_Tr

def ten(i,N):
    #index legend i = lmn, where l is whether 4 or 5 modes are considered, m designates which mode 
    #the operator is applied in, and n is 0 for anihilation op. and 1 fro creation op.
    if i == 410:
        op=tensor(destroy(N),qeye(N),qeye(N),qeye(N))
    elif i==420:
        op=tensor(qeye(N),destroy(N),qeye(N),qeye(N))
    elif i==430:    
        op=tensor(qeye(N),qeye(N),destroy(N),qeye(N))
    elif i==440:
        op=tensor(qeye(N),qeye(N),qeye(N),destroy(N))
    if i == 411:
        op=tensor(create(N),qeye(N),qeye(N),qeye(N))
    elif i==421:
        op=tensor(qeye(N),create(N),qeye(N),qeye(N))
    elif i==431:    
        op=tensor(qeye(N),qeye(N),create(N),qeye(N))
    elif i==441:
        op=tensor(qeye(N),qeye(N),qeye(N),create(N))
    elif i==510:
        op=tensor(destroy(N),qeye(N),qeye(N),qeye(N),qeye(N))
    elif i==520:
        op=tensor(qeye(N),destroy(N),qeye(N),qeye(N),qeye(N))
    elif i==530:    
        op=tensor(qeye(N),qeye(N),destroy(N),qeye(N),qeye(N))
    elif i==540:
        op=tensor(qeye(N),qeye(N),qeye(N),destroy(N),qeye(N))
    elif i==551:
        op=tensor(qeye(N),qeye(N),qeye(N),qeye(N), create(N))
    return op


# =============================================================================
# #*** Defining 2 mode states of interest ***
# 
# # State of the 2 mode split single mode squeezed vacuum using operator notation
# def tms_SMSV(N,r):
#     sq_exp = (tensor(destroy(N),qeye(N)) + tensor(qeye(N),destroy(N)))/np.sqrt(2.)
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0))
#     return Psi.unit()
#     
# 
# #*** Defining 3 mode states of interest ***
# 
# # State of the 3 mode split single mode squeezed vacuum using operator notation
# def thms_SMSV(N,r):
#     sq_exp = (tensor(destroy(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N)) + tensor(qeye(N),qeye(N),destroy(N)))/np.sqrt(3.)
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
#     
# # Three mode CV GHZ-like state 
# def GHZ3(N,r):
#     sq_a1 = (tensor(destroy(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N)) + tensor(qeye(N),qeye(N),destroy(N)))/np.sqrt(3.)
#     sq_a2 = (2*tensor(destroy(N),qeye(N),qeye(N)) - tensor(qeye(N),destroy(N),qeye(N)) - tensor(qeye(N),qeye(N),destroy(N)))/np.sqrt(6.)
#     sq_a3 = (tensor(qeye(N),destroy(N),qeye(N)) -  tensor(qeye(N),qeye(N),destroy(N)))/np.sqrt(2.)
#     Psi = squeezing(sq_a1, sq_a1, -r)*squeezing(sq_a2, sq_a2, r)*squeezing(sq_a3, sq_a3, r)*tensor(basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
# 
# #*** Defining the 4 modes states of interest ***
# # State of the 4 mode split single mode squeezed vacuum using the a-bdc bipartite splitting
# def A_BCD_SMSV(N,r):
#     sq_exp = (np.sqrt(3)*tensor(destroy(N),qeye(N)) + tensor(qeye(N),destroy(N)))/2.
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0))
#     return Psi.unit()
#     
# # State of the 4 mode split single mode squeezed vacuum using the ab-cd bipartite splitting
# def AB_CD_SMSV(N,r):
#     sq_exp = (tensor(destroy(N),qeye(N)) + tensor(qeye(N),destroy(N)))/np.sqrt(2.)
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0))
#     return Psi.unit()
#     
# # State of the 4 mode split single mode squeezed vacuum
# def fms_SMSV(N,r):
#     sq_exp = (tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N),qeye(N)) + tensor(qeye(N),qeye(N),destroy(N),qeye(N)) +  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/2.
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
# 
# # =============================================================================
# # def fms_SMSV_test(N,r):
# #     sq1 = tensor(destroy(N),qeye(N),qeye(N),qeye(N))
# #     sq2 = tensor(qeye(N),destroy(N),qeye(N),qeye(N))
# #     sq_exp = (tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N),qeye(N)) + tensor(qeye(N),qeye(N),destroy(N),qeye(N)) +  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/2.
# #     Psi = squeezing(sq1, sq1, r)*squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
# #     return Psi.unit()
# # =============================================================================
# 
# # State of the 4 mode split single mode squeezed vacuum using operator notation optimized for A-BCD
# def fms_SMSVodd(N,r):
#     sq_exp = (np.sqrt(3)*tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N),qeye(N)) + tensor(qeye(N),qeye(N),destroy(N),qeye(N)) +  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(6.)
#     Psi = squeezing(sq_exp, sq_exp, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
# 
# # Four mode CV GHZ-like state 
# def GHZ4(N,r):
#     sq_a1 = (tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N),qeye(N)) +\
#              tensor(qeye(N),qeye(N),destroy(N),qeye(N)) +  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/2.
#     sq_a2 = (3*tensor(destroy(N),qeye(N),qeye(N),qeye(N)) - tensor(qeye(N),destroy(N),qeye(N),qeye(N)) -\
#              tensor(qeye(N),qeye(N),destroy(N),qeye(N)) - tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(12.)
#     sq_a3 = (2*tensor(qeye(N),destroy(N),qeye(N),qeye(N)) - tensor(qeye(N),qeye(N),destroy(N),qeye(N)) -\
#              tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(6.)
#     sq_a4 = (tensor(qeye(N),qeye(N),destroy(N),qeye(N)) -  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(2.)
#     Psi = squeezing(sq_a1, sq_a1, -r)*squeezing(sq_a2, sq_a2, r)*squeezing(sq_a3, sq_a3, r)*\
#         squeezing(sq_a4, sq_a4, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
# 
# # The BS arrangement of A-BCD but with the source arrangement of the GHZ state 
# def A_BCD_ms(N,r):
#     sq_a1 = (3*tensor(destroy(N),qeye(N),qeye(N),qeye(N)) + tensor(qeye(N),destroy(N),qeye(N),qeye(N)) +\
#              tensor(qeye(N),qeye(N),destroy(N),qeye(N)) +  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(12.)
#     sq_a2 = (3*tensor(destroy(N),qeye(N),qeye(N),qeye(N)) - tensor(qeye(N),destroy(N),qeye(N),qeye(N)) -\
#              tensor(qeye(N),qeye(N),destroy(N),qeye(N)) - tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(12.)
#     sq_a3 = (2*tensor(qeye(N),destroy(N),qeye(N),qeye(N)) - tensor(qeye(N),qeye(N),destroy(N),qeye(N)) -\
#              tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(6.)
#     sq_a4 = (tensor(qeye(N),qeye(N),destroy(N),qeye(N)) -  tensor(qeye(N),qeye(N),qeye(N),destroy(N)))/np.sqrt(2.)
#     Psi = squeezing(sq_a2, sq_a2, r)*squeezing(sq_a3, sq_a3, r)*\
#         squeezing(sq_a4, sq_a4, r)*tensor(basis(N,0),basis(N,0),basis(N,0),basis(N,0))
#     return Psi.unit()
# 
# 
# 
# =============================================================================

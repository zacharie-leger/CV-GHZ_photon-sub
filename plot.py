# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 17:02:37 2025

@author: zacharie-leger
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
    
#*** Plots ***

lsty = [':','--','-.','--',':','-.']
marker = ['s', '^', 'P', "o", "p", "3", "4",  "8", "*" ]


titles = [r'$(AB)_{1/4}-C_{3/4}$', r'$(AB)_{1/2}-C_{1/2}$', r'$(AB)_{3/4}-C_1$', r'$Tr(D_{1/4}) (AB)_{1/4}-C_{1/4}$', r'$Tr(D_{1/4}) (AB)_{1/2}-C_{1/4}$', r'$Tr(D_{1/2}) (AB)_{1/4}-C_{1/4}$', r'$Tr((AB)_{1/4}) C_{1/2}-D_{1/4}$', r'$Tr((AB)_{1/2}) C_{1/4}-D_{1/4}$']


def logneg_vs_r(r, GHZN0ln, GHZN1ln, tot, modes, krr):
    labels = ['N='+str(modes[i])+' $k=$'+'%.2f' % krr[i] for i in range(tot)]
    col = sns.color_palette("hls", len(krr))
# =============================================================================
#     # Make figure
#     font ={#'family' : 'normal',
#             #'weight' : 'bold',
#             'size'   : 22}
#     plt.rc('font', **font)
# 
#     plt.figure(1)
# 
#     Phi0_fig1 = dict()
#     Phi1_fig1 = dict()
#     for k in range(3):
#         plt.subplot(3,1,k+1)
#         for i in range(tot):
#             Phi0_fig1[i], = plt.plot(r, GHZN0ln[k][i], color=col[i], linestyle='-', label=labels[i],linewidth=1)
#         for i in range(tot):
#             Phi1_fig1[i], = plt.plot(r, GHZN1ln[k][i], color=col[i], linestyle=lsty[i], label=labels[i],linewidth=1)
#             if k+1>1:
#                 plt.xlabel('Normalized Squeezing Parameter')
#             plt.ylabel('Log.-Neg.')
#         plt.title(titles[k])
#         
#     first_legend = plt.legend(handles=[Phi0_fig1[0],Phi0_fig1[1],Phi0_fig1[2]], loc='center right', bbox_to_anchor=(1.4, 3.25), ncol=1, title='$|\phi_0>$')
#     plt.gca().add_artist(first_legend)
#     plt.legend(handles=[Phi1_fig1[0],Phi1_fig1[1],Phi1_fig1[2]], loc='center right', bbox_to_anchor=(1.4, 1.25), ncol=1, title='$\hat{a}_A|\phi_0>$')
# 
# =============================================================================
    
    
    
    Phi0 = dict()
    Phi1 = dict()
    font ={#'family' : 'normal',
            #'weight' : 'bold',
            'size'   : 18}
    plt.rc('font', **font)
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    
    axes = axes.flatten()
    
    for k in range(8):
        
        #plt.subplot(4,2,k+1)
        for i in range(tot):
            Phi0[i], = axes[k].plot(r, GHZN0ln[k][i], color=col[i], linestyle='-', label=labels[i], linewidth=1)
        for i in range(tot):
            Phi1[i], = axes[k].plot(r, GHZN1ln[k][i], color=col[i], linestyle=lsty[i], label=labels[i], linewidth=1)
        if k+1>6:
            axes[k].set_xlabel('Squeezing $(r)$')
        if k % 2 == 0:
            axes[k].set_ylabel('Log.-Neg.')
        axes[k].set_title(titles[k])
        axes[k].grid(True)
    
    first_legend = fig.legend(handles=[Phi0[0],Phi0[1],Phi0[2]], loc='lower left', bbox_to_anchor=(0.14, -0.01), ncol=1, title=r'$|\phi_0>$')
    plt.gca().add_artist(first_legend)
    fig.legend(handles=[Phi1[0],Phi1[1],Phi1[2]], loc='lower right', bbox_to_anchor=(0.94, -0.01), ncol=1, title=r'$\hat{a}_A|\phi_0>$')
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    
    
def Gain_vs_r(r, Gain, tot, modes, krr):
    labels = ['N='+str(modes[i])+' $k=$'+'%.2f' % krr[i] for i in range(tot)]
    col = sns.color_palette("hls", len(krr))
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    axes = axes.flatten()
    for k in range(8):
        for i in range(tot):
             gain, = axes[k].plot(r, Gain[k][i], color=col[i], linestyle=lsty[i], linewidth=2)
             #print(titles[k]+str(modes[i])+'='+str(krr[np.argmax(GHZN1ln[k][i])]))
             if k==0:
                 gain.set_label(labels[i])
        if k+1>6:
            axes[k].set_xlabel('Squeezing $(r)$')
        if k % 2 == 0:
            axes[k].set_ylabel('Gain')
        axes[k].grid(True)
        
        axes[k].set_title(titles[k])
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=1)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
def Gain_vs_N(r, Gain, tot, modes, krr):
    labels = ['$k=$'+'%.2f' % krr[i] for i in range(tot)]
    col = sns.color_palette("hls", len(krr))
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    axes = axes.flatten()
    for k in range(8):
        for i in range(tot):
            if k==1:
                #axes[k].plot(np.arange(modes[0]-2,modes[-1]+1,2), Gain[k][i], color=col[i], linewidth=2)
                gain, = axes[k].plot(np.arange(modes[0]-2,modes[-1]+1,2), Gain[k][i], color=col[i], marker=marker[i], linewidth=2)
                
                gain.set_label(labels[i])
            else:
                #axes[k].plot(modes, Gain[k][i][0:len(modes)], color=col[i], linewidth=2)
                gain, = axes[k].plot(modes, Gain[k][i][0:len(modes)], color=col[i],  marker=marker[i], linewidth=2)
                
        if k+1>6:
            axes[k].set_xlabel('Number of Modes $(N)$')
        if k % 2 == 0:
            axes[k].set_ylabel('Gain')
        axes[k].grid(True)
        
        axes[k].set_title(titles[k])
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=1)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

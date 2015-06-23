# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 06:29:33 2015

@author: elf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn import decomposition

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color
    
def plot_scatter(x, y, xlabel, ylabel, n_range, lbls, title, fig_name, lgnd_name):
    fig, ax = plt.subplots()
    for i in xrange(n_range):
        ax.scatter(x[i], y[i], c=np.random.rand(3,), label=lbls[i])
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.grid();
    plt.savefig(fig_name, dpi=125)
    plt.show()    
    # create a second figure for the legend
    figLegend = plt.figure(figsize = (20,20))
    # produce a legend for the objects in the other figure
    plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left')
    # save the legend to file
    figLegend.savefig(lgnd_name)
    plt.close()

def pca_analysis( tran_ne, load_names, target_names, str_name ):
    
    pca = decomposition.PCA(n_components=4)
    pcomp = pca.fit_transform(tran_ne)
    pcomp1 = pcomp[:,0]
    pcomp2 = pcomp[:,1]
    
    N = len(target_names)
    M = len(load_names)
    load_plt = str_name + "_PCA_Load.png"
    load_plt_alt = str_name + "_PCA_Load_Alternative.png"
    load_plt_alt_lgnd = str_name + "_PCA_Load_Alternative_legend.png"
    pca_plt = str_name + "_PCA_Analysis.png"
    pca_plt_lgnd = str_name + "_PCA_Analysis_legend.png"    
    pc1_plt = str_name + "_PC1_Analysis.png" 
    pc1_plt_lgnd = str_name + "_PC1_Analysis_legend.png"
    eigen_plt = str_name + "_EigenSpectrum.png"
    
    
    #print the components value
    print pca.components_
    
    plt.scatter(
        pca.components_[0], pca.components_[1], marker = 'o')
    for label, x, y in zip(load_names,  pca.components_[0],  pca.components_[1]):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (x  - 5, y + 5),
            textcoords = 'offset points', ha = 'right', va = 'bottom', rotation=np.random.randint(0,36)*10,
            bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    plt.title("PCA Load Plot");
    plt.xlabel("effect(PC1)");
    plt.ylabel("effect(PC2)");
    plt.grid();
    plt.savefig(load_plt, dpi=125)
    plt.show()
    
    #plot loadings alternative plot
    plot_scatter(pca.components_[0], pca.components_[1], "effect(PC1)", "effect(PC2)",
                 M, load_names, "PCA Load Plot", load_plt_alt, load_plt_alt_lgnd)

    # plot the PC1&PC2 using another plot for the legend
    plot_scatter(pcomp1, pcomp2, "PC1", "PC2",
                 N, target_names, "PCA Analysis", pca_plt, pca_plt_lgnd)
    
    buckets = [0] * N
    # plot the observations in one dimension using the PC1 component and annotating the legend
    plot_scatter(pcomp1, buckets, "PC1", "",
                 N, target_names, "PCA Analysis for 1 dimension (PC1)", pc1_plt, pc1_plt_lgnd)    
   
    # print the EigenSpectrum (how much each eigenValue contributes to the final componenet), we can draw 
    # the conclusion that only the first 2 components are significant
    
    eigenValues = pca.explained_variance_ratio_
    print eigenValues
    N=len(eigenValues)
    ind = np.arange(N)
    width = 0.4
    plt.bar(ind, eigenValues,   width, color='b')
    plt.ylabel('EigenValues')
    plt.xlabel('EigenVector number')
    plt.title('EigenSpectrum')
    plt.savefig(eigen_plt, dpi=125)
    plt.show()
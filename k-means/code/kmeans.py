# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:01:38 2015

@author: elf
"""

import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def normal_func(broad_df, names, ending):
    # trying out a normality test, see how the data is distributed in the
    # broadbandmap.gov data set. Actually, I am fitting a normal curve to the data
    # and see how it differs (from the graphs is pretty clear that the data is not normaly distributed)
    for name in names:
        array = broad_df[name].values
        print name,' normality=',st.normaltest(array)
        fig_name = 'normal_test/' + ending + '_afternormalization_' + name  +'.png'
        '''
        maxVal = np.amax(array)
        minVal = np.amin(array)
        
        print 'maxVal=',maxVal,'minVal=',minVal
        
        histo = np.histogram(array, bins=100, range=(minVal, maxVal))
        freqs = histo[0]
        rangebins = (maxVal - minVal)
        numberbins = (len(histo[1])-1)
        interval = (rangebins/numberbins)
        newbins = np.arange((minVal), (maxVal), interval)
        histogram = plt.bar(newbins, freqs, width=0.01, color='gray')
        plt.show()
        plt.close()
        
        n, bins, patches = plt.hist(array, 100, normed=1)
        mu = np.mean(array)
        sigma = np.std(array)
        plt.plot(bins, mlab.normpdf(bins, mu, sigma))
        plt.title('Normal curve fit to the data');
        plt.savefig(fig_name, dpi=125)
        plt.show()
        plt.close()
        '''
        fig, ax = plt.subplots()

        # The required parameters
        num_steps = 10
        max_percentage = 0.1
        num_bins = 40
        
        # Calculating the maximum value on the y axis and the yticks
        max_val = max_percentage * len(array)
        step_size = max_val / num_steps
        yticks = [ x * step_size for x in range(0, num_steps+1) ]
        ax.set_yticks( yticks )
        plt.ylim(0, max_val)
        
        # Running the histogram method
        n, bins, patches = plt.hist(array, num_bins)
        mu = np.mean(array)
        sigma = np.std(array)
        plt.plot(bins, mlab.normpdf(bins, mu, sigma))
        
        # Before normalisation: the y axis unit is number of samples within the bin intervals in the x axis        
        # After normalisation: the y axis unit is frequency of the bin values as a percentage over all the samples
        # To plot correct percentages in the y axis     
        to_percentage = lambda y, pos: str(round( ( y / float(len(array)) ) * 100.0, 2)) + '%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))
        plt.title('Normal curve fit to the data');
        plt.savefig(fig_name, dpi=125)
        plt.show()
        plt.close()
        

def kmeans_func(broad_df, K, fileName):
    pc_toarray = broad_df.values
    hpc_fit, hpc_fit1 = train_test_split(pc_toarray, train_size=.3)
    print broad_df.head()
    
    hpc = PCA(n_components=2).fit_transform(hpc_fit)
    print hpc
    k_means = KMeans(n_clusters=K)
    k_means.fit(hpc)
    
    x_min, x_max = hpc[:, 0].min() - 8, hpc[:, 0].max() + 8
    y_min, y_max = hpc[:, 1].min() - 5, hpc[:, 1].max() + 5
    print x_min, x_max
    print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    #print 'len(xx)=',len(xx),'len(yy)=',len(yy)
    
    Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Paired,
              aspect='auto', origin='lower')
    
    plt.plot(hpc[:, 0], hpc[:, 1], 'k.', markersize=4)
    centroids = k_means.cluster_centers_
    inert = k_means.inertia_
    plt.scatter(centroids[:, 0], centroids[:, 1],
               marker='x', s=169, linewidths=3,
               color='b', zorder=8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    fileName = fileName + 'kmeans.png'
    plt.savefig(fileName, dpi=125)
    
    
    
    # Determine your k range
    k_range = range(1,14)
    
    # Fit the kmeans model for each n_clusters = k
    k_means_var = [KMeans(n_clusters=k).fit(hpc) for k in k_range]
    
    # Pull out the cluster centers for each model
    centroids = [X.cluster_centers_ for X in k_means_var]
    
    # Calculate the Euclidean distance from 
    # each point to each cluster center
    k_euclid = [cdist(hpc, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke,axis=1) for ke in k_euclid]
    
    # Total within-cluster sum of squares
    wcss = [sum(d**2) for d in dist]
    
    # The total sum of squares
    tss = sum(pdist(hpc)**2)/hpc.shape[0]
    
    # The between-cluster sum of squares
    bss = tss - wcss
    
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, bss/tss*100, 'b*-')
    ax.set_ylim((0,100))
    plt.grid(True)
    plt.xlabel('n_clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Variance Explained vs. k')
    fileName = fileName + 'variance_explained.png'
    plt.savefig(fileName, dpi=125)
    

def clustering(broad_df, target_names, K, fileName):
    #initialize and carry out clustering
    km = KMeans(n_clusters = K)
    print 'clustering ', broad_df
    km.fit(broad_df)
    
    print 'HERE'
    
    #find center of clusters
    centers = km.cluster_centers_
    print 'len(centers[0]) ', len(centers[0])
    centers[centers<0] = 0 #the minimization function may find very small negative numbers, we threshold them to 0
    centers = centers.round(2)
    centers_num = len(centers[0])
    file1 = fileName + 'centers.txt'
    f = open(file1, 'w')
    f.write('\n--------Centers of the four different clusters--------\n')
    i = 0
    county_str = 'County\t' 
    while i < K:
        j = i+1
        val = ' Cent' + str(j)
        county_str = county_str + val
        i = i + 1
    county_str = county_str + '\n'
    f.write(county_str)
    for i in range(centers_num):
        j = 0
        line = '' + target_names[i]
        while j < K:
            line = line + '\t' + str(centers[j,i])
            j = j + 1
        line = line + '\n'
        f.write(line)
    f.close()
    #find which cluster each county is in
    prediction = km.predict(broad_df)
    file2 = fileName + 'prediction.txt'
    f = open(file2, 'w')
    #f.write('--------Which cluster each county is in--------\n')
    f.write('{:<5},{}'.format('County','Cluster\n'))
    print 'len(prediction) ',len(prediction)
    for i in range(len(prediction)):
        f.write('{:<5},{}'.format(target_names[i],prediction[i]+1))
        f.write('\n')
    f.close()

def broadband_plots(normal_title, kmeansFileName):
    #normal_title = '_bband'
    broad_df = pd.read_csv('broad_data.csv', converters={'geographyId': lambda x: str(x)})
    #idx_demo = broad_df['geographyId'].str.contains('^09|^23|^25|^33|^44|^50')
    idx_demo = broad_df['geographyId'].str.contains('^10|^11|^12|^13|^24|^37|^45|^51|^54')
    broad_df = broad_df[idx_demo]
    normal_df = broad_df.copy(deep=True)
    
    del normal_df['geographyId']
    names = normal_df.T.index.values
    #print names
        
    #print broad_df
    #normal_func(broad_df, names, normal_title)
    
    broad_df.set_index('geographyId', inplace=True)
    target_names = broad_df.index.values
    #print target_names
    K = 9 #number of clusters
    #kmeansFileName = 'BBand_'
    kmeans_func(broad_df, K, kmeansFileName)
    clustering(broad_df, target_names, K, kmeansFileName)
    
def mlab_plots(normal_title, kmeansFileName):
    #normal_title = 'combined'
    df = pd.read_csv('combined_values.csv', converters={'geoid': lambda x: str(x)})
    df.set_index('geoid', inplace=True)
    #df.index.name = None
    names = df.T.index.values
    target_names = df.index.values
    #normal_func(df, names, normal_title)
    K = 9 #number of clusters
    #kmeansFileName = 'Combined_'
    df_norm = (df - df.mean()) / (df.max() - df.min())
    #print df_norm
    kmeans_func(df_norm, K, kmeansFileName)
    print target_names
    clustering(df, target_names, K, kmeansFileName)

i = 1
while i < 101:
    bband_title = 'sakmeans/BBand_'+str(i)
    comb_title = 'sakmeans/Combined_'+str(i)
    bband_normal = '_bband'
    comb_normal = '_combined'
    #broadband_plots(bband_normal, bband_title)
    mlab_plots(comb_normal, comb_title)
    i = i + 1
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:22:41 2015

@author: elf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

demo_df = pd.read_csv('density_demo_data.csv', converters={'geographyId': lambda x: str(x)})
demo_look_df = pd.read_csv('density_demo_lookup.csv', converters={'geographyId': lambda x: str(x)})


def cluster_group_print(counties, id_county, demo_df, demo_look_df, fileName):
    idx_cl = demo_df['geographyId'].str.contains(counties)
    cl_df = demo_df[idx_cl]
    cl_look_df = demo_look_df[idx_cl]
    df_name = cl_look_df[['geographyId', 'geographyName']]
    
    cl_df = pd.merge(cl_df, df_name, on='geographyId')
    cl_df.set_index('geographyName', inplace=True)
    cl_df = cl_df.drop('geographyId', 1)
    
    fig, axes = plt.subplots(nrows=len(cl_df.columns), ncols=1)
    for i, c in enumerate(cl_df.columns):
        cl_df[c].plot(kind='bar', ax=axes[i], figsize=(70, 40), title=c)
    name = fileName + 'cluster_group_' + str(id_county) + '.png'
    print name
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    

# those clusters were selected using the k-means method in kmeans.py
'''
counties = ['09003|09001|23003|25015|25025|25007|25009|25027|33009|33005|33007|33001|33003|33011|33019|33017|33015', 
            '23019|23017',
            '09007|09005|09013|09015|09011|09009|23001|23005|23015|23007|23009|23023|23011|23025|23031|23027|23029|25001|25003|25005|25021|25023|25013|25011|25017|25019|33013|44009|44001',
            '23013|23021']
'''
bband = ['23007|25003|33001|50007|50003|50001|50025|50017|50027|50015|50019|50021|50013|50023|50011',
         '44009|44001|44003|44005|44007',
         '09003|09001|09007|09005|09013|09015|09011|09009|23013|23023|25001|25005|25021|25023|25013|25011|25017|25015|25025|25007|25009|25019|33005|33011|33019|33017|33015|33013',
         '23025|23031|23027|23029',
         '23001|23003|23005|23015|23019|23017|23009|23011|25027|33009|33007|33003|50005',
         '23021|50009']


mlab = ['23001|33009|44009|33013|23031|44003|33017|09011|09013|50007|09007|25015|23019|25001|25003|09005|23011|44005',
        '09003|09009|25025|09001',
        '25017',
        '25013|25023|25005|33015|33011|23005',
        '23029|50025|23025|33019|23023|44001|23013|50005|25019|23007|23017|50017|50023|23015|50001|50011|25007|50013|50015|50027|23027|50009|33001|23021|09015|50021|23009|50019|23003|50003|25011|33005|33007|33003',
        '44007|25009|25021|25027']



bband_counties = bband
bband_fileName = 'kmeans/BBand_'

counties = mlab
fileName = 'kmeans/Combined_'

bbclusfn = 'kmeans/clusters/Density_BBand_'
combclusfn = 'kmeans/clusters/Density_Combined_'

i = 83
while i < 84:
    bb_file = bband_fileName + str(i) + 'prediction.txt'
    comb_file = fileName + str(i) + 'prediction.txt'
    bb_df = pd.read_csv(bb_file, converters={'County': lambda x: str(x)})
    comb_df = pd.read_csv(comb_file, converters={'County': lambda x: str(x)})
    j = 1
    bb_str = []
    comb_str = []
    while j < 7:
        bbs = ''
        combs = ''
        bb_df1 = bb_df.loc[bb_df['Cluster'] == j]
        comb_df1 = comb_df.loc[comb_df['Cluster'] == j]
        last_index = len(bb_df1)
        k = 0
        for index, row in bb_df1.iterrows():
            if k < last_index - 1:
                bbs = bbs + str(row['County']) + '|'
            else:
                bbs = bbs + str(row['County'])
            k = k + 1
        
        k = 0
        last_index = len(comb_df1)
        for index, row in comb_df1.iterrows():
            if k < last_index - 1:
                combs = combs + str(row['County']) + '|'
            else:
                combs = combs + str(row['County'])
            k = k + 1
        
        bb_str.append(bbs)
        comb_str.append(combs)
        j = j + 1
    bbclusfnloc = bbclusfn + str(i) + '_'
    combclusfnloc = combclusfn + str(i) + '_'
    print comb_str
    '''
    j = 0
    plt.figure(figsize=(20, 4))
    figname = 'kmeans/clusters/Median_MedianIncome_comb_'+str(i)+'.png'
    vals = []
    lbs = ['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6']
    while j < len(comb_str):
        df = pd.read_csv('demo_data.csv', converters={'geographyId': lambda x: str(x)})
        idx = df['geographyId'].str.contains(comb_str[j])
        df_sel = df[idx]
        lb = 'Cluster'+str(j)
        rtt = df_sel[['medianIncome']].median(axis=0)
        vals.append(rtt['medianIncome'])
        j = j + 1
    N = len(comb_str)
    ind = np.arange(N)
    plt.bar(ind, vals, width=1/1.5, color='steelblue')
    plt.xlabel("Number of clusters")
    plt.ylabel("Median income in $")
    plt.title("Median income by each cluster")
    plt.xticks(ind, lbs)
    plt.savefig(figname)
    plt.close()    
    

    j = 0
    plt.figure(figsize=(20, 4))
    figname = 'kmeans/clusters/Median_Median'+str(i)+'.png'
    vals = []
    lbs = ['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6']
    while j < len(comb_str):
        df = pd.read_csv('combined_values.csv', converters={'geoid': lambda x: str(x)})
        idx = df['geoid'].str.contains(comb_str[j])
        df_sel = df[idx]
        lb = 'Cluster'+str(j)
        rtt = df_sel[['MedianRTT']].median(axis=0)
        vals.append(rtt['MedianRTT'])
        j = j + 1
    N = len(comb_str)
    ind = np.arange(N)
    plt.bar(ind, vals, width=1/1.5, color='wheat')
    plt.xticks(ind, lbs)
    plt.savefig(figname)
    plt.close()
    '''
    j = 0
    plt.figure(figsize=(20, 4))
    figname = 'kmeans/clusters/MedianDownload'+str(i)+'.png'
    print len(comb_str)
    while j < len(comb_str):
        print 'j=====',j
        df = pd.read_csv('combined_values.csv', converters={'geoid': lambda x: str(x)})
        idx = df['geoid'].str.contains(comb_str[j])
        df_sel = df[idx]
        print df_sel.head()
        j = j + 1
        lb = 'Cluster'+str(j)
        df_sel['download_median'].plot(kind='line', label=lb)
    plt.legend(loc='best')
    plt.ylabel('Median Download values')
    plt.title('Median Download values for the 6 clusters')
    plt.savefig(figname)
    plt.close()

    '''
    for id_county, county in enumerate(bb_str):
        print id_county
        cluster_group_print(str(county), id_county, demo_df, demo_look_df, bbclusfnloc)
    
    for id_county, county in enumerate(comb_str):
        print id_county      
        cluster_group_print(str(county), id_county, demo_df, demo_look_df, combclusfnloc)
    '''
    i = i + 1